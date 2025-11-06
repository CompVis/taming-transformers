import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


def setup_coco_encoder():
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

    coco_ckpt = "logs/coco_epoch117.ckpt"   # path to downloaded model checkpoint file
    coco_model = VQModel(
        ddconfig=cfg.ddconfig, lossconfig=cfg.lossconfig,
        n_embed=cfg.n_embed, embed_dim=cfg.embed_dim,
        ckpt_path=coco_ckpt
    ).eval().cpu()

    return coco_model

def setup_imagenet_encoder():
    config_path = "logs/vqgan_imagenet_f16_1024/configs/model.yaml"
    ckpt_path = "logs/vqgan_imagenet_f16_1024/ckpts/last.ckpt"
    config = OmegaConf.load(config_path)
    imagenet_model = VQModel(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    imagenet_model.load_state_dict(sd, strict=False)
    imagenet_model.eval().cpu()
    return imagenet_model

MODEL = "imagenet" # "coco" or "imagenet"

if MODEL == "coco":
    encoder = setup_coco_encoder()
else:
    encoder = setup_imagenet_encoder()

codebook_raw = encoder.quantize.embedding.weight
codebook = codebook_raw.detach()

# Save codebook to file
torch.save(codebook, f'adapter/codebook-work/{MODEL}_codebook.pt')
print(f"Saved codebook of shape {codebook.shape} to adapter/codebook-work/{MODEL}_codebook.pt")