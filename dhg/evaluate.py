import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .constants import GESTURE_NAMES


def per_fold_metrics(name, y, pred, folds):
    rows = []
    for k, (_, te) in enumerate(folds, start=1):
        rows.append(dict(model=name, fold=k, acc=accuracy_score(y[te], pred[te]),
                         f1_macro=f1_score(y[te], pred[te], average="macro")))
    return rows


def summarise(per_fold: pd.DataFrame, y, oof_pred: dict, seconds: dict) -> pd.DataFrame:
    rows = []
    for name, g in per_fold.groupby("model", sort=False):
        rows.append(dict(
            model=name,
            acc_mean=g.acc.mean(), acc_std=g.acc.std(),
            f1_mean=g.f1_macro.mean(), f1_std=g.f1_macro.std(),
            pooled_acc=accuracy_score(y, oof_pred[name]),
            pooled_f1=f1_score(y, oof_pred[name], average="macro"),
            minutes=seconds.get(name, float("nan")) / 60.0,
        ))
    return pd.DataFrame(rows)


def paired_vs_baseline(per_fold: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base = per_fold[per_fold.model == baseline].set_index("fold").acc
    rows = []
    for name, g in per_fold.groupby("model", sort=False):
        if name == baseline:
            continue
        d = g.set_index("fold").acc - base
        rows.append(dict(model=name, mean_gain_pts=100 * d.mean(), folds_better=int((d > 0).sum()),
                         n_folds=len(d)))
    return pd.DataFrame(rows)


def per_class_recall(y, oof_pred: dict) -> pd.DataFrame:
    cm = {n: confusion_matrix(y, p, labels=range(14)) for n, p in oof_pred.items()}
    return pd.DataFrame({n: np.diag(c) / c.sum(1) for n, c in cm.items()}, index=GESTURE_NAMES)


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=range(14))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(14), GESTURE_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(14), GESTURE_NAMES)
    for i in range(14):
        for j in range(14):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"), ax.set_ylabel("True"), ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary.acc_mean * 100, 0.36, yerr=summary.acc_std * 100, capsize=4, label="Accuracy")
    ax.bar(x + 0.18, summary.f1_mean * 100, 0.36, yerr=summary.f1_std * 100, capsize=4, label="Macro-F1")
    ax.axhline(100 / 14, ls="--", c="gray", lw=1, label="Chance (7.1%)")
    ax.set_xticks(x, summary.model, rotation=20)
    ax.set_ylabel("Score (%)"), ax.set_title("Subject-independent 5-fold CV (mean ± std)")
    ax.legend(), ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, paired: pd.DataFrame, path, meta: dict):
    lines = ["# Results (subject-independent 5-fold GroupKFold)", ""]
    lines += [f"- Sequences: {meta['n']}, subjects: {meta['n_subjects']}, classes: 14",
              f"- Chance level: {100 / 14:.1f}%", ""]
    lines += ["| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Pooled acc | Time (min) |",
              "|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"| {r.model} | {100 * r.acc_mean:.1f} ± {100 * r.acc_std:.1f} | "
                     f"{100 * r.f1_mean:.1f} ± {100 * r.f1_std:.1f} | {100 * r.pooled_acc:.1f} | {r.minutes:.1f} |")
    if len(paired):
        lines += ["", f"## Paired comparison against `{meta['baseline']}` (same folds)", "",
                  "| Model | Mean gain (points) | Folds better |", "|---|---|---|"]
        for r in paired.itertuples():
            lines.append(f"| {r.model} | {r.mean_gain_pts:+.1f} | {r.folds_better}/{r.n_folds} |")
    lines += ["", "Std is across the 5 folds (4 held-out subjects each). With 5 folds a gain below "
              "about one std should not be called significant."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
