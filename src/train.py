import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils.funcs import load_json
from utils.logs import log

# Vision-only CLIP backbone
from model_clip import CLIPVisionOnly

# SBI dataset with CLIP normalization
from utils.sbi_clip import SBI_Dataset


# =========================
# Utils
# =========================
def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def tensor_stats(x: torch.Tensor):
    x = x.detach()
    if x.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(x.min().item()),
        float(x.max().item()),
        float(x.mean().item()),
        float(x.std(unbiased=False).item()),
    )


def is_bad_forward(logits: torch.Tensor, z: torch.Tensor) -> bool:
    if (logits is None) or (z is None):
        return True
    if (not torch.isfinite(logits).all()) or (not torch.isfinite(z).all()):
        return True
    if float(logits.abs().max().item()) == 0.0 and float(z.abs().max().item()) == 0.0:
        return True
    return False


def save_bad_batch(
    save_dir: str,
    tag: str,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor = None,
    z: torch.Tensor = None,
    extra: dict = None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{tag}.pt")
    payload = {"x": x.detach().cpu(), "y": y.detach().cpu()}
    if logits is not None:
        payload["logits"] = logits.detach().cpu()
    if z is not None:
        payload["z"] = z.detach().cpu()
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    print(f"[Saved] {path}")


def scan_nonfinite_params(model: nn.Module):
    bad = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None:
                continue
            if not torch.isfinite(p).all():
                bad.append(name)
    return bad


def scan_nonfinite_grads(model: nn.Module):
    bad = []
    for name, p in model.named_parameters():
        if p is None or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    return bad


# =========================
# FP32 LayerNorm
# =========================
class FP32LayerNorm(nn.LayerNorm):
    """
    Compute LayerNorm in FP32 and cast the output back to the input dtype.
    This improves numerical stability under mixed-precision training.
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
    """
    Recursively replace all nn.LayerNorm modules with FP32LayerNorm.
    Returns the number of replaced LayerNorm layers.
    """
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


# =========================
# Frequency-debias consistency
# =========================
def fft_lowpass(x: torch.Tensor, keep_ratio: float = 0.35) -> torch.Tensor:
    x_dtype = x.dtype
    xf = torch.fft.fft2(x.float(), dim=(-2, -1))
    xf = torch.fft.fftshift(xf, dim=(-2, -1))

    _, _, H, W = x.shape
    h = int(H * keep_ratio / 2)
    w = int(W * keep_ratio / 2)
    cy, cx = H // 2, W // 2

    mask = torch.zeros((H, W), device=x.device, dtype=torch.float32)
    mask[max(cy - h, 0):min(cy + h, H), max(cx - w, 0):min(cx + w, W)] = 1.0
    mask = mask.view(1, 1, H, W)

    xf = xf * mask
    xf = torch.fft.ifftshift(xf, dim=(-2, -1))
    y = torch.fft.ifft2(xf, dim=(-2, -1)).real
    return y.to(x_dtype)


def fft_random_debias(x: torch.Tensor) -> torch.Tensor:
    r = float(torch.empty(1, device=x.device).uniform_(0.25, 0.50).item())
    return fft_lowpass(x, keep_ratio=r)


def feature_consistency_loss(z: torch.Tensor, z_fd: torch.Tensor) -> torch.Tensor:
    z_fd = z_fd.detach()
    return 1.0 - (z * z_fd).sum(dim=1).mean()


# =========================
# PDUR: Prototype Diversity & Utilization Regularization
# =========================
def proto_diversity_loss(head: nn.Module, delta: float = 0.40) -> torch.Tensor:
    """
    Prevent prototype collapse by reducing excessive similarity
    among prototypes within each class.
    """
    P = F.normalize(head.prototypes.float(), dim=-1)  # (C,K,D)
    C, K, _ = P.shape
    if K <= 1:
        return P.new_tensor(0.0)

    loss = P.new_tensor(0.0)
    for c in range(C):
        G = P[c] @ P[c].t()  # (K,K)
        G = G - torch.eye(K, device=G.device, dtype=G.dtype)
        loss = loss + F.relu(G - delta).mean()
    return loss / C


def proto_balance_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    head: nn.Module,
    tau: float = 0.12,
) -> torch.Tensor:
    """
    Prevent prototype starvation by encouraging more balanced
    responsibility assignment over prototypes of the same class.
    """
    P = F.normalize(head.prototypes.float(), dim=-1)  # (C,K,D)
    zf = z.float()
    cos_all = torch.einsum("nd,ckd->nck", zf, P)  # (N,C,K)

    C = P.size(0)
    K = P.size(1)

    loss = z.new_tensor(0.0)
    cnt = 0
    for c in range(C):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        cos_ck = cos_all[idx, c, :]                     # (Nc,K)
        r = F.softmax(cos_ck / max(tau, 1e-6), dim=1)   # (Nc,K)
        r_bar = r.mean(dim=0)                           # (K,)

        u = torch.full_like(r_bar, 1.0 / K)
        loss = loss + (r_bar * (torch.log(r_bar + 1e-8) - torch.log(u))).sum()
        cnt += 1

    return loss / max(cnt, 1)


# =========================
# LR schedule
# =========================
def mpam_lr(
    epoch: int,
    step: int,
    steps_per_epoch: int,
    min_lr: float = 1e-5,
    max_lr: float = 3e-4,
    cycle_len: int = 10,
) -> float:
    epoch_in_cycle = epoch % cycle_len
    t = (step + 1) / max(1, steps_per_epoch)
    if epoch_in_cycle == 0:
        return min_lr + (max_lr - min_lr) * t
    progress = (epoch_in_cycle - 1 + t) / (cycle_len - 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =========================
# Resolve snapshot dir
# =========================
def resolve_snapshot_dir(repo_dir: str) -> str:
    snap_root = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"Repo dir not found: {repo_dir}")
    if not os.path.isdir(snap_root):
        raise FileNotFoundError(f"Snapshots dir not found: {snap_root}")

    snaps = [d for d in os.listdir(snap_root) if os.path.isdir(os.path.join(snap_root, d))]
    if not snaps:
        raise FileNotFoundError(f"No snapshot folders found in: {snap_root}")

    snaps.sort()
    for name in reversed(snaps):
        cand = os.path.join(snap_root, name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    raise FileNotFoundError(f"No valid snapshot contains config.json under: {snap_root}")


# =========================
# Head: Multi-Prototype + Adaptive Margin
# =========================
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


# =========================
# Model: LN-tuning + BN + MPAM head
# =========================
class MPAM_CLIP(nn.Module):
    def __init__(self, clip_vision: CLIPVisionOnly, head_cfg: dict = None):
        super().__init__()
        self.backbone = clip_vision
        feat_dim = clip_vision.get_features_dim()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self._ln_cnt = replace_layernorm_with_fp32(self.backbone)
        for m in self.backbone.modules():
            if isinstance(m, FP32LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        self.bn = nn.BatchNorm1d(
            feat_dim,
            affine=False,
            track_running_stats=True,
            momentum=0.01,
            eps=1e-5,
        )

        head_cfg = head_cfg or {}
        self.classifier = MultiProtoAdaptiveMarginHead(
            in_dim=feat_dim,
            num_classes=2,
            n_proto=int(head_cfg.get("n_proto", 5)),
            s=float(head_cfg.get("s", 30.0)),
            m0=float(head_cfg.get("m0", 0.20)),
            lam=float(head_cfg.get("lam", 0.30)),
            m_min=float(head_cfg.get("m_min", 0.0)),
            m_max=float(head_cfg.get("m_max", 0.60)),
            margin_type=str(head_cfg.get("margin_type", "arc")).lower(),
            learnable_alpha=bool(head_cfg.get("learnable_alpha", True)),
        ).float()

        for p in self.classifier.parameters():
            p.requires_grad = True

    def _bn_forward(self, feat_fp32: torch.Tensor, update_stats: bool) -> torch.Tensor:
        if update_stats:
            return self.bn(feat_fp32)

        return F.batch_norm(
            feat_fp32,
            self.bn.running_mean,
            self.bn.running_var,
            weight=None,
            bias=None,
            training=False,
            momentum=0.0,
            eps=self.bn.eps,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, bn_update: bool = True):
        feat = self.backbone(x).float()
        feat = self._bn_forward(feat, bn_update)
        z = F.normalize(feat, dim=-1)
        logits = self.classifier(z, y)
        return logits, z


# =========================
# Train
# =========================
def main(args):
    cfg = load_json(args.config)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    n_epoch = int(cfg.get("epoch", 15))
    n_frames = int(cfg.get("n_frames", 8))

    gamma = float(cfg.get("gamma", 0.05))

    mu_div = float(cfg.get("mu_div", 0.01))
    mu_bal = float(cfg.get("mu_bal", 0.02))
    delta_div = float(cfg.get("delta_div", 0.40))
    tau_bal = float(cfg.get("tau_bal", 0.12))

    train_dataset = SBI_Dataset(phase="train", image_size=image_size, n_frames=n_frames)
    val_dataset = SBI_Dataset(phase="val", image_size=image_size, n_frames=n_frames)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    hub_repo_dir = cfg.get(
        "hub_repo_dir",
        "/root/autodl-tmp/src/hf_home/hub/models--openai--clip-vit-large-patch14",
    )
    snapshot_dir = cfg.get("snapshot_dir", None) or resolve_snapshot_dir(hub_repo_dir)
    clip_vision = CLIPVisionOnly(local_dir=snapshot_dir, return_projection=False)

    head_cfg = cfg.get("head", {})
    model = MPAM_CLIP(clip_vision, head_cfg=head_cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0)

    now = datetime.now()
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    timestamp = now.strftime("%m_%d_%H_%M_%S")
    save_path = f"output/{args.session_name}_{config_stem}_{timestamp}/"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + "weights/", exist_ok=True)
    os.makedirs(save_path + "logs/", exist_ok=True)
    debug_dir = os.path.join(save_path, "debug_bad_batches")
    os.makedirs(debug_dir, exist_ok=True)

    logger = log(path=save_path + "logs/", file="losses.logs")
    logger.info(
        f"Device={device} | image_size={image_size} | batch(2B)={batch_size} | "
        f"loader_batch={batch_size//2} | "
        f"trainable_params={sum(p.numel() for p in trainable_params)} | fp32_ln_replaced={model._ln_cnt} | "
        f"hub_repo_dir={hub_repo_dir} | snapshot_dir={snapshot_dir} | "
        f"gamma={gamma} | PDUR(mu_div={mu_div}, mu_bal={mu_bal}, delta={delta_div}, tau={tau_bal}) | "
        f"head={head_cfg}"
    )

    best_auc = -1.0
    keep_topk = int(cfg.get("keep_topk", 5))
    weight_dict = {}

    steps_per_epoch = len(train_loader)

    for epoch in range(n_epoch):
        model.train(True)
        train_loss = 0.0
        train_acc = 0.0
        avg_ce = 0.0
        avg_cons = 0.0
        avg_div = 0.0
        avg_bal = 0.0

        for step, data in enumerate(tqdm(train_loader, desc=f"Train {epoch + 1}/{n_epoch}")):
            x = data["img"].to(device, non_blocking=True)
            y = data["label"].to(device, non_blocking=True).long()

            if device.type == "cuda":
                x = x.half()

            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            lr = mpam_lr(epoch, step, steps_per_epoch, min_lr=1e-5, max_lr=3e-4, cycle_len=10)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            logits, z = model(x, y, bn_update=True)

            use_cons = (torch.rand(1, device=device).item() < 0.5) and (gamma > 0.0)
            if use_cons:
                x_fd = fft_random_debias(x)
                logits_fd, z_fd = model(x_fd, None, bn_update=False)
            else:
                logits_fd, z_fd = None, None

            if is_bad_forward(logits, z):
                ls = tensor_stats(logits)
                zs = tensor_stats(z)
                print(f"[Bad forward] epoch={epoch} step={step} logits_stats={ls} z_stats={zs}")
                save_bad_batch(
                    debug_dir,
                    tag=f"badfwd_e{epoch}_s{step}",
                    x=x,
                    y=y,
                    logits=logits,
                    z=z,
                    extra={"logits_stats": ls, "z_stats": zs, "lr": lr},
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            if use_cons and is_bad_forward(logits_fd, z_fd):
                ls = tensor_stats(logits_fd)
                zs = tensor_stats(z_fd)
                print(f"[Bad forward FD] epoch={epoch} step={step} logits_stats={ls} z_stats={zs}")
                save_bad_batch(
                    debug_dir,
                    tag=f"badfwd_fd_e{epoch}_s{step}",
                    x=x,
                    y=y,
                    logits=logits_fd,
                    z=z_fd,
                    extra={"logits_stats": ls, "z_stats": zs, "lr": lr},
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            logits_fp32 = logits.float()
            z_fp32 = z.float()

            ce = F.cross_entropy(logits_fp32, y)

            if use_cons:
                cons = feature_consistency_loss(z_fp32, z_fd.float())
            else:
                cons = z_fp32.new_tensor(0.0)

            div = proto_diversity_loss(model.classifier, delta=delta_div) if (mu_div > 0) else z_fp32.new_tensor(0.0)
            bal = proto_balance_loss(z_fp32, y, model.classifier, tau=tau_bal) if (mu_bal > 0) else z_fp32.new_tensor(0.0)

            loss = ce + gamma * cons + mu_div * div + mu_bal * bal

            if not torch.isfinite(loss):
                print(f"[Bad loss] epoch={epoch} step={step} loss={float(loss.item())}")
                save_bad_batch(
                    debug_dir,
                    tag=f"badloss_e{epoch}_s{step}",
                    x=x,
                    y=y,
                    logits=logits_fp32,
                    z=z_fp32,
                    extra={
                        "ce": float(ce.item()) if torch.isfinite(ce) else None,
                        "cons": float(cons.item()) if torch.isfinite(cons) else None,
                        "div": float(div.item()) if torch.isfinite(div) else None,
                        "bal": float(bal.item()) if torch.isfinite(bal) else None,
                        "lr": lr,
                        "use_cons": bool(use_cons),
                    },
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            loss.backward()

            bad_grads = scan_nonfinite_grads(model)
            if bad_grads:
                print(f"[Bad grad] epoch={epoch} step={step} bad_grads_count={len(bad_grads)}")
                save_bad_batch(
                    debug_dir,
                    tag=f"badgrad_e{epoch}_s{step}",
                    x=x,
                    y=y,
                    logits=logits_fp32,
                    z=z_fp32,
                    extra={"bad_grads": bad_grads, "lr": lr},
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            optimizer.step()

            bad_params = scan_nonfinite_params(model)
            if bad_params:
                print(f"[Bad param] epoch={epoch} step={step} bad_params_count={len(bad_params)}")
                save_bad_batch(
                    debug_dir,
                    tag=f"badparam_e{epoch}_s{step}",
                    x=x,
                    y=y,
                    logits=logits_fp32,
                    z=z_fp32,
                    extra={"bad_params": bad_params, "lr": lr},
                )
                raise RuntimeError(f"Non-finite params detected: {bad_params[:10]} ...")

            train_loss += float(loss.detach().item())
            train_acc += compute_accuracy(logits_fp32.detach(), y)

            avg_ce += float(ce.detach().item())
            avg_cons += float(cons.detach().item())
            avg_div += float(div.detach().item())
            avg_bal += float(bal.detach().item())

            if device.type == "cuda" and (step % 50 == 0):
                torch.cuda.empty_cache()

        denom = max(1, len(train_loader))
        train_loss /= denom
        train_acc /= denom
        avg_ce /= denom
        avg_cons /= denom
        avg_div /= denom
        avg_bal /= denom

        model.train(False)
        val_loss = 0.0
        val_acc = 0.0
        probs = []
        gts = []

        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc=f"Val {epoch + 1}/{n_epoch}")):
                x = data["img"].to(device, non_blocking=True)
                y = data["label"].to(device, non_blocking=True).long()

                if device.type == "cuda":
                    x = x.half()
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

                logits_eval, z = model(x, None, bn_update=False)

                if is_bad_forward(logits_eval, z):
                    print(f"[Bad val forward] epoch={epoch} step={step}")
                    save_bad_batch(
                        debug_dir,
                        tag=f"badvalfwd_e{epoch}_s{step}",
                        x=x,
                        y=y,
                        logits=logits_eval,
                        z=z,
                        extra={"lr": None},
                    )
                    continue

                logits_eval_fp32 = logits_eval.float()
                z_fp32 = z.float()

                logits_margin_fp32 = model.classifier(z_fp32, y).float()
                ce = F.cross_entropy(logits_margin_fp32, y)
                loss = ce

                if not torch.isfinite(loss):
                    print(f"[Bad val loss] epoch={epoch} step={step}")
                    continue

                val_loss += float(loss.item())
                val_acc += compute_accuracy(logits_eval_fp32, y)

                p_fake = logits_eval_fp32.softmax(dim=1)[:, 1].detach().cpu().numpy()
                probs.extend(p_fake.tolist())
                gts.extend(y.detach().cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_acc /= max(1, len(val_loader))

        probs_np = np.asarray(probs, dtype=np.float64)
        gts_np = np.asarray(gts, dtype=np.int64)

        m = np.isfinite(probs_np)
        probs_np = probs_np[m]
        gts_np = gts_np[m]

        if probs_np.size == 0 or len(set(gts_np.tolist())) <= 1:
            val_auc = 0.0
        else:
            val_auc = float(roc_auc_score(gts_np, probs_np))

        log_text = (
            f"Epoch {epoch + 1}/{n_epoch} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"train ce {avg_ce:.4f} cons {avg_cons:.4f} div {avg_div:.4f} bal {avg_bal:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} | "
            f"gamma {gamma} mu_div {mu_div} mu_bal {mu_bal}"
        )
        logger.info(log_text)

        ckpt_path = os.path.join(save_path, "weights", f"{epoch + 1}_{val_auc:.4f}_val.tar")

        def save_ckpt(path):
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "val_auc": float(val_auc),
                    "snapshot_dir": snapshot_dir,
                },
                path,
            )

        if len(weight_dict) < keep_topk:
            save_ckpt(ckpt_path)
            weight_dict[ckpt_path] = val_auc
        else:
            worst_path = min(weight_dict, key=lambda k: weight_dict[k])
            worst_auc = weight_dict[worst_path]
            if val_auc >= worst_auc:
                try:
                    os.remove(worst_path)
                except Exception:
                    pass
                del weight_dict[worst_path]
                save_ckpt(ckpt_path)
                weight_dict[ckpt_path] = val_auc

        best_auc = max(best_auc, val_auc)

    logger.info(f"Done. Best val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="config")
    parser.add_argument("-n", dest="session_name", default="mpam_clip")
    args = parser.parse_args()
    main(args)