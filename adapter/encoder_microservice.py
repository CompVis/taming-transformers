from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import base64, io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from torchvision.utils import save_image

# Run this command to start the server:
# uvicorn adapter.encoder_microservice:app --host 0.0.0.0 --port 8000


class LegacyEncoder:
    def __init__(self, checkpoint_path=None):
        # init model/checkpoints pinned to old Python
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

        coco_model = VQModel(
            ddconfig=cfg.ddconfig, lossconfig=cfg.lossconfig,
            n_embed=cfg.n_embed, embed_dim=cfg.embed_dim,
            ckpt_path=checkpoint_path
        ).eval().cpu()

        self.model = coco_model


    def preprocess(self, img_pil: Image.Image):
        # VQGAN expects images in [-1,1]
        prep = T.Compose([
            T.Resize(256, interpolation=Image.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)  # maps [0,1] -> [-1,1]
        ])

        return prep(img_pil.convert("RGB")).unsqueeze(0)
    

    def encode(self, img):
        # return latent as torch.Tensor
        with torch.no_grad():
            latent, _, _ = self.model.encode(img)
        return latent
    

    def decode(self, latent):
        # return decoded letent
        with torch.no_grad():
            recon = self.model.decode(latent)
        return recon
    
    def denorm(self, x):
        return (x.clamp(-1, 1) + 1) / 2.0

enc = LegacyEncoder(checkpoint_path="logs/coco_epoch117.ckpt")
app = FastAPI()

class LatentPayload(BaseModel):
    # A class designed to carry string-encoded latent arrays
    # across HTTP requests
    latent_b64: str

def _to_b64_npy(arr: np.ndarray) -> str:
    """
    Encode to base 64 for transport as a plain string
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _from_b64_npy(s: str) -> np.ndarray:
    """
    Decode a transported string back to array
    """
    raw = base64.b64decode(s.encode("ascii"))
    return np.load(io.BytesIO(raw), allow_pickle=False)

def _to_b64_tsr(tensor: torch.Tensor) -> str:
    """
    Encode a PyTorch tensor to base64 for transport as a plain string.
    """
    buf = io.BytesIO()
    # Always move to CPU before serializing (safer for cross-device transport)
    torch.save(tensor.cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _from_b64_tsr(s: str) -> torch.Tensor:
    """
    Decode a base64 string back into a PyTorch tensor.
    """
    raw = base64.b64decode(s.encode("utf-8"))
    buf = io.BytesIO(raw)
    return torch.load(buf) 

@app.post("/encode")
async def encode_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data))
    x = enc.preprocess(img)      
    z = enc.encode(x).detach().cpu().numpy()
    return {"latent_b64": _to_b64_npy(z)}

@app.post("/decode")
async def decode_endpoint(payload: LatentPayload):
    z = _from_b64_npy(payload.latent_b64)   
    z_t = torch.from_numpy(z)              
    x = enc.decode(z_t)
    x = enc.denorm(x) 
    out = io.BytesIO()
    save_image(x, out, format="PNG")
    return Response(content=out.getvalue(), media_type="image/png")