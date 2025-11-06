import os
import sys
import time
import platform
import argparse

import json
from collections import Counter

import torch
import numpy as np
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from omegaconf import OmegaConf
import PIL
from PIL import Image
from taming.models.vqgan import VQModel, GumbelVQ
from tqdm import tqdm

# Ensure local imports work
sys.path.append(".")

# Disable gradients to save memory
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def load_config(config_path, display=False):
	config = OmegaConf.load(config_path)
	if display:
		print(yaml.dump(OmegaConf.to_container(config)))
	return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
	if is_gumbel:
		model = GumbelVQ(**config.model.params)
	else:
		model = VQModel(**config.model.params)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def norm01(img):
		img = (img + 1.0) / 2.0
		return img

def preprocess(img, target_image_size=256):
    s = min(img.size)  # shortest side

    # Just allow upscaling instead of raising
    # (for larger images this will downscale; for smaller, it will upscale)
    r = target_image_size / s
    new_h = round(r * img.size[1])
    new_w = round(r * img.size[0])

    img = TF.resize(img, (new_h, new_w), interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img

@torch.no_grad()
def swap_codewords(
	model,                    # taming VQModel (eval mode)
	z_e,                      # [B, D, Hq, Wq] from encoder/quant_conv
	indices,             # [B*Hq*Wq] or [Hq*Wq] flattened indices
	old_id: int,
	new_id: int):

	# infer shapes
	B, D, Hq, Wq = z_e.shape
	e_dim = model.quantize.e_dim
	n_embed = model.quantize.n_e
	assert 0 <= old_id < n_embed and 0 <= new_id < n_embed, "index out of range"

	# swap old → new
	device = indices.device
	idx_swapped = torch.where(
		indices == old_id,
		torch.as_tensor(new_id, device=device, dtype=indices.dtype),
		indices
	)

	# rebuild z_q from edited indices
	z_q_edit = model.quantize.get_codebook_entry(idx_swapped, (B, Hq, Wq, e_dim))  # -> [B, D, Hq, Wq]

	return z_q_edit, idx_swapped


def load_neighbours(path, near=True):
	# Load nearest codewords
	data = np.load(path, allow_pickle=True)

	neighbors = {
		"indices_near": data["idx_near"],
		"distances_near": data["dist_near"],
		"indices_far": data["idx_far"],
		"distances_far": data["dist_far"],
		"metric": str(data["metric"]),
		"k": int(data["k"]),
	}
	if near:
		return neighbors["indices_near"], neighbors["distances_near"], neighbors["metric"]
	else:
		return neighbors["indices_far"], neighbors["distances_far"], neighbors["metric"]


@torch.no_grad()
def generate_images(img_path, model, out_dir_name, do_knn=True, do_kfn=True):

	# Document generated images
	# Key: Path to original image
	# Value: the image metadat of the form:
	"""
	{
	"original_image": True | False,
	"reconstructed_only": True | False,
    "swap_type": "furthest" | "nearest" | "",
    "num_swaps": int: Number of codeword swaps performed
	}
	"""
	generated_images = {}

	# Preprocess image
	img = Image.open(img_path).convert("RGB")
	x_vqgan = preprocess(img)
	x = preprocess_vqgan(x_vqgan).to(DEVICE)
	z_q, _, [_, _, indices] = model.encode(x)

	# Save undoctored reconstruction
	rf = norm01(model.decode(z_q))
	path = f"{out_dir_name}/reconstructed_no_pert.png"
	save_image(rf, path)

	log = {"original_image": False,
			"reconstructed_only": True,
        	"swap_type": "furthest",
        	"num_swaps": 0}
	generated_images[path] = log
	
	# Find most and least prominent codewords
	codeword_counter = Counter(indices.cpu().numpy().tolist())
	sorted_codeword_counts = codeword_counter.most_common()
	
	running_pert_closest = z_q.clone()
	running_pert_furthest = z_q.clone()

	# keep running indices in sync with the running latents
	cur_idx_closest = indices.clone()
	cur_idx_furthest = indices.clone()
	
	if do_knn:
		# load nearest codewords
		idxs_near, dists_near, metric_near = load_neighbours(CODEWORD_NEIGHBOURS_PATH, near=True)

		for i in range(K_NEAREST):
			old_id = sorted_codeword_counts[i][0]

			# pick a true neighbor (skip self)
			nn_list = idxs_near[old_id]
			new_close_id = nn_list[1] if nn_list[0] == old_id else nn_list[0]

			# cumulative/running swaps (update both latent and indices each time)
			running_pert_closest, cur_idx_closest = swap_codewords(model, running_pert_closest, cur_idx_closest, old_id=old_id, new_id=new_close_id)

			# decode and save
			rc = norm01(model.decode(running_pert_closest))

			# Save images
			path = f"{out_dir_name}/ns_{i}_most_prominent.png"
			save_image(rc, path)
			log = {"original_image": False,
				"reconstructed_only": False,
				"swap_type": "nearest",
				"num_swaps": i + 1}
			generated_images[path] = log

	if do_kfn:   
		# load furthest codewords
		idxs_furthest, dists_furthest, metric_furthest = load_neighbours(CODEWORD_NEIGHBOURS_PATH, near=False)
		for i in range(K_FURTHEST):
			old_id = sorted_codeword_counts[i][0]

			# pick a true neighbor (skip self)
			fn_list = idxs_furthest[old_id]
			new_far_id   = fn_list[0]  # furthest list shouldn't include self, but ok if it does

			# cumulative/running swaps (update both latent and indices each time)
			running_pert_furthest, cur_idx_furthest = swap_codewords(model, running_pert_furthest, cur_idx_furthest, old_id=old_id, new_id=new_far_id)

			# decode and save
			rf = norm01(model.decode(running_pert_furthest))
			path = f"{out_dir_name}/fs_{i}_most_prominent.png"
			save_image(rf, path)

			log = {"original_image": False,
				"reconstructed_only": False,
				"swap_type": "furthest",
				"num_swaps": i + 1}
			generated_images[path] = log
	return generated_images


def load_image_caption_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":	
	# ------------------------------------------------------------------------------
	# Set up command line argument parser
	parser = argparse.ArgumentParser(description="Run the model configuration")	
	parser.add_argument("--model", type=str, default="imagenet",
	                    choices=["imagenet", "coco"],
	                    help="Model type to use (imagenet or coco)")	
	parser.add_argument("--display_config", action="store_true",
	                    help="Display configuration info (default: False)")	
	parser.add_argument("--max_images", type=int, default=2,
	                    help="Maximum number of images to process")	
	parser.add_argument("--k_nearest", type=int, default=20,
	                    help="Number of nearest neighbours to use")	
	parser.add_argument("--k_furthest", type=int, default=20,
	                    help="Number of furthest neighbours to use")	
	parser.add_argument("--output_path", type=str, default="outputs",
	                    help="Output path for generated results")	
	
	# Parse arguments from the command line
	args = parser.parse_args()

	# Assign to variables for easier reference
	MODEL = args.model
	DISPLAY_CONFIG = args.display_config
	MAX_IMAGES = args.max_images
	K_NEAREST = args.k_nearest
	K_FURTHEST = args.k_furthest
	OUTPUT_PATH = f"adapter/demo/{args.output_path}"	
	# ------------------------------------------------------------------------------
	# Codeword neighbours are generated in the LatentLab repository in codeword_nearest_neighbours.ipynb.
	if MODEL == "imagenet":
		IMAGE_DATA_PATH = "inputs/zsc/validation_images_and_labels_500.json"
		CODEWORD_NEIGHBOURS_PATH = "adapter/perturbing/imagenet-codeword-files/imagenet_neighbors_data.npz"
	elif MODEL == "coco":
		CODEWORD_NEIGHBOURS_PATH = "adapter/perturbing/coco-codeword-files/coco_neighbors_data.npz"
		IMAGE_DATA_PATH = "inputs/itm/images_and_captions.json"
	else:
		raise ValueError("Unsupported MODEL type. Choose 'coco' or 'imagenet'.")
	#------------------------------------------------------------------------------

	model = None

	print(f"Setting up {MODEL} VQ-GAN Model")
	start_time = time.time()
	if MODEL == "imagenet":
		config_path = "logs/vqgan_imagenet_f16_1024/configs/model.yaml"
		ckpt_path = "logs/vqgan_imagenet_f16_1024/ckpts/last.ckpt"
		config = load_config(config_path, display=DISPLAY_CONFIG)
		model = load_vqgan(config, ckpt_path=ckpt_path).to(DEVICE)

		# Load images and labels
		images_and_labels = load_image_caption_dict(IMAGE_DATA_PATH)
		image_paths = images_and_labels.keys()

	elif MODEL == "coco":
		ckpt_path = "logs/coco_epoch117.ckpt"

		cfg = OmegaConf.create({
		"ddconfig": {
			"double_z": False, "z_channels": 256, "resolution": 256,
			"in_channels": 3, "out_ch": 3, "ch": 128,
			"ch_mult": [1,1,2,2,4], "num_res_blocks": 2,
			"attn_resolutions": [16], "dropout": 0.0
		},
		"lossconfig": {"target":"taming.modules.losses.DummyLoss"},
		"n_embed": 8192, "embed_dim": 256
		})

		if DISPLAY_CONFIG:
			print(cfg)

		model = VQModel(
			ddconfig=cfg.ddconfig, lossconfig=cfg.lossconfig,
			n_embed=cfg.n_embed, embed_dim=cfg.embed_dim,
			ckpt_path=ckpt_path
		).eval().to(DEVICE)

		# Load images and captions
		images_and_captions = load_image_caption_dict(IMAGE_DATA_PATH)
		image_paths = images_and_captions.keys()
	
	###########################################################################
	# Define new models + checkpoints here
	###########################################################################
	else:
		raise ValueError("Unsupported MODEL type. Choose 'coco' or 'imagenet'.")

	# Preprocessing transform
	prep = T.Compose([
		T.Resize(256, interpolation=Image.BICUBIC),
		T.CenterCrop(256),
		T.ToTensor(),
	])

	dir_name = f"{OUTPUT_PATH}/perturbed-{MODEL}"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	print(len(image_paths), "images found.")

	generated_images = {}

	for count, img_path in enumerate(tqdm(list(image_paths)[:MAX_IMAGES], desc="Processing images")):
		img_name = os.path.basename(img_path).split('.')[0]

		if MODEL == "imagenet":
			class_id = os.path.basename(os.path.dirname(img_path)) # Last folder is image class ID
			subdir_name = f"{dir_name}/{class_id}/{img_name}"
		else:
			subdir_name = f"{dir_name}/{img_name}"
		if os.path.exists(subdir_name):
			# Count files (ignore subdirectories)
			files = [f for f in os.listdir(subdir_name) if os.path.isfile(os.path.join(subdir_name, f))]
			if len(files) == (K_NEAREST + K_FURTHEST + 1):  # knn + kfn + 1 direct reconstruction
				tqdm.write(f"Skipping {img_name}, directory already exists with expected number of files ({len(files)}).")
				continue
		os.makedirs(subdir_name, exist_ok=True)

		# Optional: use tqdm.write to avoid progress bar interference
		tqdm.write(f"Processing image: {img_name}")

		# Generate perturbed images
		img_generated_images = generate_images(img_path, model, subdir_name, do_knn=True, do_kfn=True)
		generated_images[img_path] = img_generated_images
	

	# Save generated images info
	with open(f"{dir_name}/generated_images.json", 'w', encoding='utf-8') as f:
		json.dump(generated_images, f, ensure_ascii=False, indent=4)
	
	end_time = time.time()
	images_generated = len(generated_images) * (K_NEAREST + K_FURTHEST + 1)
	time_taken = end_time - start_time
	time_per_image = 0 if images_generated == 0 else time_taken / images_generated

	print(images_generated, f" images generated in {time_taken:.4f} seconds.")
	print(f"An average of {time_per_image:.4f} seconds per perturbed image generated")

	print("Running on:", platform.system(), platform.release())
	print("Processor:", platform.processor())
	print("Architecture:", platform.architecture())
	print("CPU cores:", os.cpu_count())
