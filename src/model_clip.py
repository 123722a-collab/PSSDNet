# src/model_clip.py
import os

# ======= Fixed hub repo cache directory (do not change) =======
HUB_REPO_DIR = "/root/autodl-tmp/src/hf_home/hub/models--openai--clip-vit-large-patch14"

# (Optional but recommended) Point Hugging Face environment variables
# to the parent cache directories so transformers will look here first.
HF_HUB_CACHE = os.path.dirname(HUB_REPO_DIR)  # /root/autodl-tmp/src/hf_home/hub
HF_HOME = os.path.dirname(HF_HUB_CACHE)       # /root/autodl-tmp/src/hf_home
os.makedirs(HF_HUB_CACHE, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ========= imports =========
import torch
import torch.nn as nn
from transformers import CLIPModel


def resolve_snapshot_dir(repo_dir: str) -> str:
    """
    repo_dir: /.../hub/models--openai--clip-vit-large-patch14
    return:   /.../hub/models--openai--clip-vit-large-patch14/snapshots/<hash>
    """
    snap_root = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Repo dir not found: {repo_dir}")
    if not os.path.isdir(snap_root):
        raise FileNotFoundError(f"Snapshots dir not found: {snap_root}")

    snaps = [d for d in os.listdir(snap_root) if os.path.isdir(os.path.join(snap_root, d))]
    if not snaps:
        raise FileNotFoundError(f"No snapshot folders found in: {snap_root}")

    snaps.sort()
    # Select the latest snapshot that contains config.json
    for name in reversed(snaps):
        cand = os.path.join(snap_root, name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand

    raise FileNotFoundError(f"No valid snapshot contains config.json under: {snap_root}")


class CLIPVisionOnly(nn.Module):
    """
    Vision-only CLIP ViT-L/14 interface (Transformers), loaded offline from a local snapshot directory.

    Input:
      pixel_values: (B, 3, 224, 224)

    Output:
      - return_projection=False: pooled hidden features, shape (B, 1024)
      - return_projection=True : projected embedding, shape (B, projection_dim)
    """

    def __init__(self, local_dir: str, return_projection: bool = False, freeze_vision: bool = False):
        super().__init__()
        self.return_projection = bool(return_projection)

        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"local_dir is not a directory: {local_dir}")
        if not os.path.isfile(os.path.join(local_dir, "config.json")):
            raise FileNotFoundError(f"config.json not found in: {local_dir}")

        clip: CLIPModel = CLIPModel.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )

        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.features_dim = int(self.vision_model.config.hidden_size)  # Usually 1024 for ViT-L/14

        if freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.vision_model(pixel_values).pooler_output  # (B, 1024)
        if self.return_projection:
            return self.visual_projection(pooled)
        return pooled

    def get_features_dim(self) -> int:
        return self.features_dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snapshot_dir = resolve_snapshot_dir(HUB_REPO_DIR)
    model = CLIPVisionOnly(local_dir=snapshot_dir, return_projection=False).to(device).eval()

    B = 2
    dtype = next(model.parameters()).dtype
    x = torch.randn(B, 3, 224, 224, device=device, dtype=dtype)

    with torch.no_grad():
        y = model(x)

    print("✅ CLIP ViT-L/14 vision-only loaded and forward pass succeeded")
    print(f"  HUB_REPO_DIR   : {HUB_REPO_DIR}")
    print(f"  snapshot_dir   : {snapshot_dir}")
    print(f"  HF_HOME        : {HF_HOME}")
    print(f"  HF_HUB_CACHE   : {HF_HUB_CACHE}")
    print(f"  device         : {device}")
    print(f"  model dtype    : {dtype}")
    print(f"  input shape    : {tuple(x.shape)}")
    print(f"  output shape   : {tuple(y.shape)}")
    print(f"  features_dim   : {model.get_features_dim()}")
    print(f"  output mean/std: {y.float().mean().item():.4f} / {y.float().std().item():.4f}")


if __name__ == "__main__":
    main()