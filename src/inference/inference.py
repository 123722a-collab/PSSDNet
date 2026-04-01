import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from model_clip import CLIPVisionOnly


# ============================
# Face crop cache
# ============================
CACHE_ROOT = "./face_cache"


def get_faces_with_cache(filename, num_frames, face_detector, dataset_name, cache_root=CACHE_ROOT):
    video_id = os.path.splitext(os.path.basename(filename))[0]
    cache_dir = os.path.join(cache_root, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{video_id}_{num_frames}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        faces = data["faces"]
        idxs = data["idxs"]
        return faces, idxs

    faces, idxs = extract_frames(filename, num_frames, face_detector)
    faces_arr = np.array(faces)
    idxs_arr = np.array(idxs)
    np.savez_compressed(cache_path, faces=faces_arr, idxs=idxs_arr)
    return faces_arr, idxs_arr


# ============================
# CLIP normalization
# ============================
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_normalize(x01: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(x01.device, dtype=x01.dtype)
    std = CLIP_STD.to(x01.device, dtype=x01.dtype)
    return (x01 - mean) / std


# ============================
# FP32 LayerNorm
# ============================
class FP32LayerNorm(nn.LayerNorm):
    """
    Compute LayerNorm in FP32 and cast the output back to the input dtype.
    This matches the training implementation under mixed precision.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        y = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return y.to(orig_dtype)


def replace_layernorm_with_fp32(module: nn.Module) -> int:
    cnt = 0
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            new_ln = FP32LayerNorm(
                child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
            )
            if child.elementwise_affine:
                new_ln.weight.data.copy_(child.weight.data.float())
                new_ln.bias.data.copy_(child.bias.data.float())
            setattr(module, name, new_ln)
            cnt += 1
        else:
            cnt += replace_layernorm_with_fp32(child)
    return cnt


# ============================
# Head: Multi-Prototype + Adaptive Margin
# In inference, margin is disabled by setting y=None
# ============================
class MultiProtoAdaptiveMarginHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        n_proto: int = 5,
        s: float = 30.0,
        m0: float = 0.20,
        lam: float = 0.30,
        m_min: float = 0.0,
        m_max: float = 0.60,
        margin_type: str = "arc",
        learnable_alpha: bool = True,
        eps_acos: float = 1e-6,
    ):
        super().__init__()
        assert margin_type in ["arc", "cos"]
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.n_proto = int(n_proto)
        self.s = float(s)
        self.m0 = float(m0)
        self.lam = float(lam)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.margin_type = margin_type
        self.eps_acos = float(eps_acos)

        self.prototypes = nn.Parameter(torch.empty(num_classes, self.n_proto, in_dim))
        nn.init.xavier_uniform_(self.prototypes)

        if learnable_alpha:
            self.alpha_logits = nn.Parameter(torch.zeros(num_classes, self.n_proto))
        else:
            self.register_buffer("alpha_logits", torch.zeros(num_classes, self.n_proto))

    def _alpha(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits.float(), dim=1)

    def forward(self, z: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        P = F.normalize(self.prototypes.float(), dim=-1)
        zf = z.float()
        cos_all = torch.einsum("nd,ckd->nck", zf, P)
        cos_all = torch.clamp(cos_all, -1.0 + self.eps_acos, 1.0 - self.eps_acos)

        alpha = self._alpha()
        cos_bar = (cos_all * alpha.unsqueeze(0)).sum(dim=2)

        if y is None:
            return cos_bar * self.s

        cos_bar_y = cos_bar.gather(1, y.view(-1, 1)).squeeze(1)
        m_i = self.m0 + self.lam * (1.0 - cos_bar_y)
        m_i = torch.clamp(m_i, self.m_min, self.m_max)

        N = z.shape[0]
        cos_mod = cos_all.clone()
        idx = y.view(-1, 1, 1).expand(N, 1, self.n_proto)
        cos_yk = cos_all.gather(1, idx).squeeze(1)

        if self.margin_type == "arc":
            theta_yk = torch.acos(torch.clamp(cos_yk, -1.0 + self.eps_acos, 1.0 - self.eps_acos))
            cos_yk_m = torch.cos(theta_yk + m_i.view(N, 1))
        else:
            cos_yk_m = cos_yk - m_i.view(N, 1)

        cos_mod.scatter_(1, idx, cos_yk_m.unsqueeze(1))
        cos_bar_mod = (cos_mod * alpha.unsqueeze(0)).sum(dim=2)
        return cos_bar_mod * self.s


# ============================
# Normalize head cfg from ckpt
# ============================
def normalize_head_cfg(head_cfg: dict) -> dict:
    head_cfg = dict(head_cfg or {})

    if "n_proto" not in head_cfg:
        if "num_prototypes" in head_cfg:
            head_cfg["n_proto"] = head_cfg["num_prototypes"]
        elif "K" in head_cfg:
            head_cfg["n_proto"] = head_cfg["K"]

    if "lam" not in head_cfg and "lambda_margin" in head_cfg:
        head_cfg["lam"] = head_cfg["lambda_margin"]

    head_cfg.setdefault("n_proto", 5)
    head_cfg.setdefault("s", 30.0)
    head_cfg.setdefault("m0", 0.20)
    head_cfg.setdefault("lam", 0.30)
    head_cfg.setdefault("m_min", 0.0)
    head_cfg.setdefault("m_max", 0.60)
    head_cfg.setdefault("margin_type", "arc")
    head_cfg.setdefault("learnable_alpha", True)

    return head_cfg


# ============================
# MPAM_CLIP wrapper (inference)
# ============================
class MPAM_CLIP(nn.Module):
    def __init__(self, clip_vision: CLIPVisionOnly, head_cfg: dict):
        super().__init__()
        self.backbone = clip_vision
        feat_dim = clip_vision.get_features_dim()

        head_cfg = normalize_head_cfg(head_cfg)

        self.classifier = MultiProtoAdaptiveMarginHead(
            in_dim=feat_dim,
            num_classes=2,
            n_proto=int(head_cfg["n_proto"]),
            s=float(head_cfg["s"]),
            m0=float(head_cfg["m0"]),
            lam=float(head_cfg["lam"]),
            m_min=float(head_cfg["m_min"]),
            m_max=float(head_cfg["m_max"]),
            margin_type=str(head_cfg["margin_type"]).lower(),
            learnable_alpha=bool(head_cfg["learnable_alpha"]),
        ).float()

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x).float()
        z = F.normalize(feat, dim=-1)
        logits = self.classifier(z, None)
        return logits, z


# ============================
# Load checkpoint
# ============================
def load_ckpt_and_build_model(ckpt_path: str, clip_vision: CLIPVisionOnly, device: torch.device):
    assert os.path.isfile(ckpt_path), f"[ERROR] Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)

    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise RuntimeError("[ERROR] Invalid checkpoint format: expected a dict containing key='model'.")

    cfg = ckpt.get("cfg", {})
    head_cfg_raw = cfg.get("head", {})
    head_cfg = normalize_head_cfg(head_cfg_raw)

    model = MPAM_CLIP(clip_vision, head_cfg=head_cfg).to(device)

    sd = ckpt["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("bn.")}
    if "classifier.prototypes" in sd:
        k_ckpt = int(sd["classifier.prototypes"].shape[1])
        k_build = int(model.classifier.n_proto)
        if k_ckpt != k_build:
            raise RuntimeError(
                f"[ERROR] Inconsistent n_proto: ckpt={k_ckpt}, built_model={k_build}\n"
                f"  ckpt head_cfg(raw)={head_cfg_raw}\n"
                f"  head_cfg(normalized)={head_cfg}\n"
                f"  Please make sure training and inference use the same n_proto."
            )

    info = model.load_state_dict(sd, strict=False)
    missing, unexpected = info.missing_keys, info.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"[ERROR] Checkpoint and model do not match\n"
            f"  missing: {missing}\n"
            f"  unexpected: {unexpected}\n"
            f"  ckpt head_cfg(raw): {head_cfg_raw}\n"
            f"  head_cfg(normalized): {head_cfg}\n"
        )

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] head_cfg(raw): {head_cfg_raw}")
    print(f"[INFO] head_cfg(normalized): {head_cfg}")
    return model


# ============================
# Video-level aggregation
# ============================
def aggregate_video_score(
    pred_fake: torch.Tensor,
    idx_list: np.ndarray,
    topk_ratio: float = 0.2,
    topk_min: int = 8,
) -> float:
    """
    Top-k mean aggregation:
      1) group by frame index, using max score within each frame
      2) select top-k frame scores
      3) average the selected scores
    """
    pred = pred_fake.detach().cpu().numpy()

    frame_scores = []
    idx_img = -1
    cur = []
    for i in range(len(pred)):
        idx = int(idx_list[i])
        if idx != idx_img:
            if len(cur) > 0:
                frame_scores.append(float(max(cur)))
            cur = []
            idx_img = idx
        cur.append(float(pred[i]))
    if len(cur) > 0:
        frame_scores.append(float(max(cur)))

    if len(frame_scores) == 0:
        return 0.5

    fs = np.asarray(frame_scores, dtype=np.float32)
    k = int(round(len(fs) * float(topk_ratio)))
    k = max(int(topk_min), k)
    k = min(len(fs), k)

    topk = np.partition(fs, -k)[-k:]
    return float(topk.mean())


# ============================
# EER
# ============================
def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = 0.5 * (fpr[idx] + fnr[idx])
    eer_th = thresholds[idx]

    if idx + 1 < len(fpr):
        d1 = fpr[idx] - fnr[idx]
        d2 = fpr[idx + 1] - fnr[idx + 1]
        if d1 * d2 < 0:
            t = d1 / (d1 - d2)
            fpr_i = fpr[idx] + t * (fpr[idx + 1] - fpr[idx])
            fnr_i = fnr[idx] + t * (fnr[idx + 1] - fnr[idx])
            th_i = thresholds[idx] + t * (thresholds[idx + 1] - thresholds[idx])
            eer_i = 0.5 * (fpr_i + fnr_i)
            return float(eer_i), float(th_i)

    return float(eer), float(eer_th)


# ============================
# Main
# ============================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snapshot_dir = args.clip_snapshot_dir
    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(f"[ERROR] clip_snapshot_dir not found: {snapshot_dir}")
    if not os.path.isfile(os.path.join(snapshot_dir, "config.json")):
        raise FileNotFoundError(f"[ERROR] config.json not found in: {snapshot_dir}")

    clip_vision = CLIPVisionOnly(local_dir=snapshot_dir, return_projection=False).to(device)
    replace_layernorm_with_fp32(clip_vision)

    model = load_ckpt_and_build_model(args.weight_name, clip_vision, device)
    model.eval()
    model.backbone.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    if args.dataset == "FFIW":
        video_list, target_list = init_ffiw()
    elif args.dataset == "FF":
        video_list, target_list = init_ff()
    elif args.dataset == "DFD":
        video_list, target_list = init_dfd()
    elif args.dataset == "DFDC":
        video_list, target_list = init_dfdc()
    elif args.dataset == "DFDCP":
        video_list, target_list = init_dfdcp()
    elif args.dataset == "CDF":
        video_list, target_list = init_cdf()
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    output_list = []
    for filename in tqdm(video_list):
        try:
            faces, idxs = get_faces_with_cache(filename, args.n_frames, face_detector, args.dataset)

            with torch.no_grad():
                x = torch.tensor(faces, device=device).float()
                if x.numel() == 0:
                    raise RuntimeError("empty faces")
                if x.max() > 1.5:
                    x = x / 255.0

                x = clip_normalize(x)

                logits, _ = model(x)
                pred_fake = logits.softmax(dim=1)[:, 1]

            pred_video = aggregate_video_score(
                pred_fake,
                idxs,
                topk_ratio=float(args.topk_ratio),
                topk_min=int(args.topk_min),
            )

        except Exception as e:
            print(f"[Warn] failed on {filename}: {e}")
            pred_video = 0.5

        output_list.append(pred_video)

    y_true = np.asarray(target_list, dtype=np.int32)
    y_score = np.asarray(output_list, dtype=np.float64)

    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    eer, eer_th = compute_eer(y_true, y_score)

    print(f"{args.dataset} | AUC: {auc:.4f} | AP: {ap:.4f} | EER: {eer:.4f} (th={eer_th:.6f})")


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_name", type=str, required=True, help="Path to the trained checkpoint (.tar)")
    parser.add_argument("-d", dest="dataset", type=str, required=True, help="Dataset name: FF / DFDC / CDF / ...")
    parser.add_argument("-n", dest="n_frames", default=32, type=int)

    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=0.2,
        help="Top-k ratio for video-level aggregation (e.g., 0.2 means top 20%)",
    )
    parser.add_argument(
        "--topk_min",
        type=int,
        default=8,
        help="Minimum number of selected frames for video-level aggregation",
    )

    parser.add_argument(
        "--clip_snapshot_dir",
        type=str,
        required=True,
        help="e.g. /root/autodl-tmp/src/hf_home/hub/models--openai--clip-vit-large-patch14/snapshots/<hash>",
    )

    args = parser.parse_args()
    main(args)
