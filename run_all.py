"""Run the full comparison on DHG-2016 with subject-independent cross-validation.

    python run_all.py --data /path/to/DHG2016 --out results
    python run_all.py --data /path/to/DHG2016 --models pose_svm gru --epochs 60 --seeds 3

The dataset path can also come from the DHG_DATASET_PATH environment variable.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dhg import evaluate as ev
from dhg.data import load_dataset, normalize, quality_report
from dhg.features import all_classical
from dhg.train_dl import DLConfig, run_dl_cv

CLASSICAL = {"pose_svm": "pose"}
DEEP = ("gru",)
ALL_MODELS = list(CLASSICAL) + list(DEEP)


def build_sklearn(n_features, quick):
    """Scaler -> SelectKBest -> RBF-SVM. Scaler and selection are fitted inside each training fold only."""
    pipe = Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif)),
                     ("svm", SVC(kernel="rbf", gamma="scale"))])
    ks, cs = [300, 600], [10, 100]
    grid = {"select__k": [min(ks[0], n_features)] if quick else [min(k, n_features) for k in ks],
            "svm__C": cs[:1] if quick else cs}
    return pipe, grid


def run_sklearn(F, y, groups, folds, quick, n_jobs, log):
    pred = np.full(len(y), -1)
    t0 = time.time()
    for k, (tr, te) in enumerate(folds, start=1):
        pipe, grid = build_sklearn(F.shape[1], quick)
        gs = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(n_splits=3), scoring="f1_macro", n_jobs=n_jobs)
        gs.fit(F[tr], y[tr], groups=groups[tr])
        pred[te] = gs.predict(F[te])
        log(f"    fold {k}/{len(folds)} acc {(pred[te] == y[te]).mean():.4f}  best {gs.best_params_}")
    return pred, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.getenv("DHG_DATASET_PATH", "./data/DHG2016"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache/dhg_sequences.npz")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=1, help="seeds averaged per fold for the GRU")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2, help="parallel jobs for the SVM grid search")
    ap.add_argument("--quick", action="store_true", help="tiny grids / few epochs, for smoke tests")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    data = load_dataset(args.data, cache=args.cache)
    Xn, scale = normalize(data["X"])
    quality = quality_report(data["X"], scale)
    y, groups = data["y"], data["subject"]
    log(f"{len(y)} sequences, {len(np.unique(groups))} subjects, {len(np.unique(y))} classes")

    folds = list(GroupKFold(n_splits=args.folds).split(Xn, y, groups))
    fold_of = np.zeros(len(y), dtype=int)
    for k, (_, te) in enumerate(folds, start=1):
        fold_of[te] = k
    pd.DataFrame(dict(index=range(len(y)), subject=groups, gesture=y, fold=fold_of)).to_csv(out / "folds.csv", index=False)

    feats = all_classical(Xn) if any(m in CLASSICAL for m in args.models) else {}
    if feats:
        log("Feature sizes: " + ", ".join(f"{k}={v.shape[1]}" for k, v in feats.items()))

    cfg = DLConfig(epochs=2 if args.quick else args.epochs, seeds=args.seeds, threads=args.threads)
    preds, probs, seconds, dl_info = {}, {}, {}, {}
    for name in args.models:
        log(f"=== {name} ===")
        if name in CLASSICAL:
            preds[name], seconds[name] = run_sklearn(feats[CLASSICAL[name]], y, groups, folds,
                                                     args.quick, args.n_jobs, log)
        else:
            probs[name], seconds[name], dl_info[name] = run_dl_cv(name, Xn, y, groups, folds, cfg, log)
            preds[name] = probs[name].argmax(1)
        log(f"  pooled accuracy: {(preds[name] == y).mean():.4f}   ({seconds[name] / 60:.1f} min)")

    rows = []
    for name, p in preds.items():
        rows += ev.per_fold_metrics(name, y, p, folds)
    per_fold = pd.DataFrame(rows)
    summary = ev.summarise(per_fold, y, preds, seconds)
    baseline = "pose_svm" if "pose_svm" in preds else next(iter(preds))
    paired = ev.paired_vs_baseline(per_fold, baseline)

    per_fold.to_csv(out / "per_fold.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    ev.per_class_recall(y, preds).to_csv(out / "per_class_recall.csv")
    np.savez_compressed(out / "oof_predictions.npz", y=y, subject=groups, **{f"pred_{k}": v for k, v in preds.items()},
                        **{f"prob_{k}": v for k, v in probs.items()})
    ev.plot_comparison(summary, out / "comparison.png")
    for name, p in preds.items():
        ev.plot_confusion(y, p, f"{name} - confusion matrix (all folds pooled)", out / f"confusion_{name}.png")
    ev.write_markdown(summary, paired, out / "summary.md",
                      dict(n=len(y), n_subjects=len(np.unique(groups)), baseline=baseline))
    (out / "run_info.json").write_text(json.dumps(
        dict(args=vars(args), quality=quality, dl_epoch_info=dl_info, folds=args.folds), indent=2, default=str))

    log("\n" + summary.assign(acc=lambda d: (100 * d.acc_mean).round(1), f1=lambda d: (100 * d.f1_mean).round(1))
        [["model", "acc", "f1", "minutes"]].to_string(index=False))
    log(f"\nAll outputs written to {out}/")
    return summary


if __name__ == "__main__":
    main()
