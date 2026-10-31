# Results (subject-independent 5-fold GroupKFold)

- Sequences: 2800, subjects: 20, classes: 14
- Chance level: 7.1%

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Pooled acc | Time (min) |
|---|---|---|---|---|
| pose_svm | 72.7 ± 5.8 | 72.8 ± 5.3 | 72.7 | 1.6 |
| gru | 84.7 ± 4.9 | 84.7 ± 4.7 | 84.7 | 53.3 |

## Paired comparison against `pose_svm` (same folds)

| Model | Mean gain (points) | Folds better |
|---|---|---|
| gru | +12.0 | 5/5 |

Std is across the 5 folds (4 held-out subjects each). With 5 folds a gain below about one std should not be called significant.
