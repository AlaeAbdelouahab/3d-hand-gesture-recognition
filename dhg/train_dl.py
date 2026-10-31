"""Subject-wise cross-validated training of the GRU model."""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_CLASSES
from .models import build_model


@dataclass
class DLConfig:
    epochs: int = 80
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    seeds: int = 1
    n_val_subjects: int = 2
    augment: bool = True
    threads: int = 0          # 0 = torch default


def make_channels(Xn: torch.Tensor) -> torch.Tensor:
    """(B, T, V, 3) normalised coordinates -> (B, T, V, 9) channels:
    palm-centred pose (3) | palm trajectory from the first frame, broadcast to all joints (3)
    | frame-to-frame velocity (3)."""
    palm = Xn[:, :, 1:2, :]
    rel = Xn - palm
    traj = (palm - palm[:, :1]).expand_as(Xn)
    vel = torch.diff(Xn, dim=1, prepend=Xn[:, :1])
    return torch.cat([rel, traj, vel], dim=-1)


def _rotation_matrices(B: int, max_deg: float, gen: torch.Generator) -> torch.Tensor:
    ang = (torch.rand(B, 3, generator=gen) * 2 - 1) * math.radians(max_deg)
    cx, cy, cz = torch.cos(ang).unbind(1)
    sx, sy, sz = torch.sin(ang).unbind(1)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Training-only augmentation on normalised coordinates (B, T, V, 3):
    small 3D rotation about the starting palm, amplitude scaling, jitter, and a random
    temporal crop re-interpolated to the original length (speed variation)."""
    B, T, V, _ = x.shape
    c = x[:, :1, 1:2, :]                                           # start palm position
    R = _rotation_matrices(B, 12.0, gen)
    x = torch.einsum("btvi,bji->btvj", x - c, R) + c
    scale = 0.9 + 0.2 * torch.rand(B, 1, 1, 1, generator=gen)
    x = (x - c) * scale + c
    x = x + 0.02 * torch.randn(x.shape, generator=gen)
    out = torch.empty_like(x)
    for b in range(B):
        frac = 0.8 + 0.2 * torch.rand(1, generator=gen).item()
        n = max(4, int(round(T * frac)))
        start = int(torch.randint(0, T - n + 1, (1,), generator=gen).item())
        seg = x[b, start:start + n].reshape(n, V * 3).T.unsqueeze(0)      # (1, V*3, n)
        seg = F.interpolate(seg, size=T, mode="linear", align_corners=True)
        out[b] = seg.squeeze(0).T.reshape(T, V, 3)
    return out


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(model(make_channels(X[i:i + bs])), dim=1))
    return torch.cat(out).numpy()


def train_one(kind, Xtr, ytr, Xva, yva, cfg: DLConfig, seed: int, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = build_model(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(Xtr) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    best_acc, best_state, best_ep = -1.0, None, -1

    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2:
                continue
            xb = Xtr[idx]
            if cfg.augment:
                xb = augment(xb, gen)
            loss = F.cross_entropy(model(make_channels(xb)), ytr[idx],
                                   label_smoothing=cfg.label_smoothing)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        val_acc = float((predict_proba(model, Xva).argmax(1) == yva.numpy()).mean())
        if val_acc >= best_acc:
            best_acc, best_ep = val_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"      epoch {ep + 1:3d}/{cfg.epochs}  loss {loss.item():.3f}  val_acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    return model, best_acc, best_ep


def run_dl_cv(kind, Xn, y, groups, folds, cfg: DLConfig, log=print):
    """Returns (oof probabilities (N, 14), seconds, per-fold info list)."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    X = torch.from_numpy(Xn)
    yt = torch.from_numpy(y).long()
    oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    info, t0 = [], time.time()
    for k, (tr, te) in enumerate(folds):
        rng = np.random.RandomState(1000 + k)
        subs = np.unique(groups[tr])
        val_subs = rng.choice(subs, size=min(cfg.n_val_subjects, len(subs) - 1), replace=False)
        va_mask = np.isin(groups[tr], val_subs)
        tr_in, va_in = tr[~va_mask], tr[va_mask]
        probs = np.zeros((len(te), NUM_CLASSES), dtype=np.float32)
        for s in range(cfg.seeds):
            log(f"    fold {k + 1}/{len(folds)} seed {s}  (train {len(tr_in)}, val {len(va_in)}, test {len(te)})")
            model, vacc, vep = train_one(kind, X[tr_in], yt[tr_in], X[va_in], yt[va_in], cfg, 17 * s + k, log)
            probs += predict_proba(model, X[te])
            info.append(dict(fold=k + 1, seed=s, val_acc=vacc, best_epoch=vep))
        oof[te] = probs / cfg.seeds
        acc = float((oof[te].argmax(1) == y[te]).mean())
        log(f"    fold {k + 1} test accuracy: {acc:.4f}")
    return oof, time.time() - t0, info
