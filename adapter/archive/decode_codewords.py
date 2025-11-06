import os
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from taming.models.vqgan import VQModel

"""
Author: Gil Wise

This script loads a COCO-trained VQGAN model and visualizes its learned codebook embeddings.
It reconstructs minimal and tiled image patches from individual codewords, helping to interpret
what each latent vector “draws.” 

Main components:
- setup_encoder(): Builds the VQModel using COCO parameters and checkpoint weights.
- decode_codeword_patch(): Decodes a single codeword (1x1 latent) into an image patch.
- decode_tiled(): Repeats a codeword across an HxW latent grid to visualize its texture.
- cosine_sims(): Prints codeword norms and cosine similarity samples for inspection.
- new(): Saves visualizations of smallest/largest-norm codewords and side-by-side comparisons.
- old(): Decodes all (or a subset of) codewords and saves them as PNG images.

Use this script to explore and understand the structure of the VQGAN latent space.
"""


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_encoder():
    # Load the same params as in coco_scene_images_transformer.yaml:
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

    coco_ckpt = "logs/coco_epoch117.ckpt"   # <— path you downloaded
    model = VQModel(
        ddconfig=cfg.ddconfig, lossconfig=cfg.lossconfig,
        n_embed=cfg.n_embed, embed_dim=cfg.embed_dim,
        ckpt_path=coco_ckpt
    )

    sd = torch.load(coco_ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)

    model = model.to(DEVICE).eval()

    return model

@torch.no_grad()
def decode_codeword_patch(model, e_k):
    """
    Decode a minimal image patch for codeword index k by feeding a 1x1 latent token.
    Returns a float tensor in [0,1], shape [1, 3, H_out, W_out].
    """
    # Make a 1x1 latent grid using that vector: [B=1, D, H=1, W=1]
    zq = e_k.view(1, -1, 1, 1).to(DEVICE)

    # Decode: output is typically in [-1, 1]
    x_hat = model.decode(zq).clamp(-1, 1)

    # Map to [0,1] for saving as image
    x_hat = (x_hat + 1) / 2.0
    return x_hat

@torch.no_grad()
def decode_tiled(model, e_k, hw=8):
    zq = e_k.view(1, -1, 1, 1).to(DEVICE)

    # Scale up to hw x hw by repeating
    zq = zq.expand(1, zq.shape[1], hw, hw)

    x = model.decode(zq).clamp(-1, 1)
    return (x + 1) / 2.0

# -------------------------------------------
# Output directory
# -------------------------------------------
out_dir = "adapter/codebook-work/coco/decoded-codebook-patches"
os.makedirs(out_dir, exist_ok=True)


# -------------------------------------------
# Model setup
# -------------------------------------------
vqgan = setup_encoder()

# -------------------------------------------
# Load codebook, decode all codewords and save as images
# -------------------------------------------

def cosine_sims():
    with torch.no_grad():
        codebook = vqgan.quantize.embedding.weight.detach()

        # norms & pairwise similarity sample
        norms = codebook.norm(dim=1)
        print("min/max codeword norm:", norms.min().item(), norms.max().item())
        # sample cosine similarities against a few anchors

        anchors = torch.randint(0, codebook.shape[0], (16,))
        S = F.cosine_similarity(codebook[anchors][:,None,:], codebook[None,:,:], dim=-1)  # [16, K]
        print("anchor mean cos-sim:", S.mean(dim=1))

def new():
    E = vqgan.quantize.embedding.weight.detach()
    norms = E.norm(dim=1)
    K = E.shape[0]

    # pick N smallest & N largest norms
    N = 10
    small_idx = torch.topk(norms, N, largest=False).indices
    large_idx = torch.topk(norms, N, largest=True).indices

    os.makedirs("viz_small_norm", exist_ok=True)
    os.makedirs("viz_large_norm", exist_ok=True)

    with torch.no_grad():
        for i, k in enumerate(small_idx.tolist()):
            patch = decode_tiled(vqgan, E[k], hw=8)
            save_image(patch, f"viz_small_norm/code_{k:04d}_n{norms[k]:.3f}.png")

        for i, k in enumerate(large_idx.tolist()):
            patch = decode_tiled(vqgan, E[k], hw=8)
            save_image(patch, f"viz_large_norm/code_{k:04d}_n{norms[k]:.3f}.png")


    def concat_side_by_side(a, b):  # both [1,3,H,W] in [0,1]
        H = max(a.shape[-2], b.shape[-2])
        a = F.interpolate(a, (H, H))
        b = F.interpolate(b, (H, H))
        return torch.cat([a, b], dim=-1)

    k = torch.randint(0, K, ()).item()
    patch_1x1 = (vqgan.decode(E[k].view(1,-1,1,1).to(DEVICE)).clamp(-1,1)+1)/2
    patch_8x8 = decode_tiled(vqgan, E[k], hw=8)
    save_image(concat_side_by_side(patch_1x1, patch_8x8), f"compare_{k:04d}.png")

def old():
    with torch.no_grad():
        codebook = vqgan.quantize.embedding.weight
        n_codewords = codebook.shape[0]

        for k in tqdm(range(1), desc="Decoding codewords"):
            e_k = codebook[k]
            print(f"Codeword shape: {e_k.shape}")
            patch = decode_codeword_patch(vqgan, e_k)  # [1, 3, H, W]
    
            save_path = os.path.join(out_dir, f"codeword_{k:04d}.png")
            save_image(patch, save_path)

    print(f"Decoded and saved {n_codewords} codeword patches to {out_dir}")


# Un-comment one of both of the following:
#old()
#new()