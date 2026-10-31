"""Hand-crafted features for the classical (SVM) models. All inputs are normalised
sequences Xn of shape (N, T, 22, 3) from `data.normalize`."""

import numpy as np

from .constants import SUBGRAPH_NODES


def _pairwise_stats(P: np.ndarray) -> dict:
    """P: (N, T, k, 3) node positions of a subgraph -> per-frame statistics (N, T)."""
    k = P.shape[2]
    d = np.linalg.norm(P[:, :, :, None, :] - P[:, :, None, :, :], axis=-1)
    iu = np.triu_indices(k, 1)
    dd = d[:, :, iu[0], iu[1]]
    cen = P.mean(axis=2, keepdims=True)
    return dict(
        mean=dd.mean(-1), std=dd.std(-1), max=dd.max(-1),
        cent=np.linalg.norm(P - cen, axis=-1).mean(axis=2),   # compactness
        disp=P.var(axis=2).mean(axis=-1),                      # 3D dispersion
    )


def graph_features(Xn: np.ndarray):
    """Subgraph descriptors of the original report (vectorised, ~465 features):
    per frame and per subgraph: mean/std/max pairwise distance, centroid distance,
    dispersion; plus speed/acceleration/delta of the mean distance and the palm
    displacement (move_X/Y/Z)."""
    feats, names = [], []
    T = Xn.shape[1]
    palm = Xn[:, :, 1, :]
    for part, nodes in SUBGRAPH_NODES.items():
        st = _pairwise_stats(Xn[:, :, nodes, :])
        for key in ("mean", "std", "max", "cent", "disp"):
            feats.append(st[key])
            names += [f"{part}_{key}_F{t}" for t in range(T)]
        dm = np.diff(st["mean"], axis=1)
        feats.append(np.stack([
            dm.mean(1), np.abs(dm).max(1), np.diff(dm, axis=1).mean(1),
            st["mean"][:, -1] - st["mean"][:, 0],
        ], axis=1))
        names += [f"{part}_{n}" for n in ("speed_mean", "speed_max", "acc_mean", "delta_global")]
    feats.append(palm[:, -1] - palm[:, 0])
    names += ["move_X", "move_Y", "move_Z"]
    return np.concatenate(feats, axis=1).astype(np.float32), names


def pose_features(Xn: np.ndarray, n_segments: int = 5, n_traj: int = 10):
    """Features the original descriptors throw away: where each joint is relative to
    the palm (hand shape *and* orientation), how it evolves over time (segment means,
    motion energy) and the palm trajectory (direction, path length, per-third motion)."""
    N, T, V, _ = Xn.shape
    palm = Xn[:, :, 1, :]
    rel = Xn - palm[:, :, None, :]

    segs = np.array_split(np.arange(T), n_segments)
    seg_means = np.stack([rel[:, s].mean(1) for s in segs], axis=1)      # (N, S, V, 3)
    motion = np.abs(np.diff(rel, axis=1)).mean(1)                         # (N, V, 3)

    traj = palm - palm[:, :1]
    idx = np.linspace(0, T - 1, n_traj).round().astype(int)
    pv = np.diff(palm, axis=1)
    speed = np.linalg.norm(pv, axis=-1)
    thirds = np.array_split(np.arange(T), 3)
    third_disp = np.stack([palm[:, s[-1]] - palm[:, s[0]] for s in thirds], axis=1)
    extra = np.concatenate([
        traj[:, -1], speed.sum(1, keepdims=True), speed.max(1, keepdims=True), pv.mean(1),
    ], axis=1)

    f = np.concatenate([
        seg_means.reshape(N, -1), motion.reshape(N, -1),
        traj[:, idx].reshape(N, -1), third_disp.reshape(N, -1), extra,
    ], axis=1)
    return f.astype(np.float32)


def all_classical(Xn: np.ndarray) -> dict:
    """Input of the classical model: the original graph descriptors + pose/trajectory features."""
    g, _ = graph_features(Xn)
    p = pose_features(Xn)
    return {"pose": np.concatenate([g, p], axis=1)}
