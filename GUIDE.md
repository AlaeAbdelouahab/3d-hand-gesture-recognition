# Guide: how to run it and how to understand it

## 1. Running it (Windows)

```bash
cd dhg-gesture-recognition      # important: run from this folder, otherwise you get "No module named dhg"
pip install -r requirements.txt
python run_all.py --out results --epochs 60
```

- Put the dataset in `data/DHG2016/`. If it is elsewhere, add `--data "path\to\DHG2016"`.
- The first run reads the 2,800 text files, which takes a few minutes, and saves them in `cache/dhg_sequences.npz`. Later runs skip that step.
- To run only the SVM (a couple of minutes), add `--models pose_svm`. For only the GRU, use `--models gru` (10 to 50 minutes depending on your CPU).
- You will see many "Features are constant" warnings. They are harmless: 21 of the 908 features are always zero (for example the palm position relative to itself), and the feature selection ignores them.
- Results land in `results/`: tables, confusion matrices and per-fold scores.
- Before pushing to GitHub, check that `.gitignore` is doing its job: the dataset, the cache and the large `.npz` files should not be uploaded.

## 2. The story in five sentences
1. DHG-2016 has 14 gestures performed by 20 people, and every frame gives 22 joints with (x, y, z) coordinates.
2. I always test on people the model has never seen, which is the honest way to measure it.
3. My course version (distance statistics on hand subgraphs + SVM) reached 32.9%, because those statistics lose direction and orientation.
4. Adding pose and trajectory features brought the SVM to 72.7%.
5. A bidirectional GRU that reads the frames directly reaches 84.7%, and beats the SVM on all 5 folds.

## 3. What to look up, in order of importance

**Must understand**
- Evaluating by subject (GroupKFold), and why a random split would leak information and inflate the score
- Data leakage, and why the scaler and the feature selection live inside the pipeline
- Accuracy vs macro-F1, the confusion matrix, and chance level (1/14 = 7.1%)
- Normalising by hand size, and resampling every sequence to the same length
- Feature engineering (hand-made features) vs letting a network learn its own
- SVM with an RBF kernel, the parameters C and gamma, and why you standardize first
- RNN, then the vanishing gradient problem, then GRU (update and reset gates); bidirectional; pooling over time
- Backpropagation (through time), gradient descent, mini-batches and epochs

**Good to understand**
- Cross-validation, the inner cross-validation used for hyperparameters, grid search
- SelectKBest with f_classif (ranking features with an ANOVA F-test)
- Overfitting, dropout, weight decay (AdamW), picking the best epoch on a validation set
- Data augmentation, and why I don't mirror the hand
- Learning-rate schedule (OneCycleLR), label smoothing, gradient clipping, BatchNorm

**Nice to know**
- The hand seen as a graph (joints are nodes, bones are edges), subgraphs, distance to the centroid, dispersion
- Mean ± standard deviation over folds, and why a gain smaller than one standard deviation proves nothing
- Why a single seed limits what I can claim, and how my protocol differs from the official DHG split

## 4. What each file does, and why it exists

### At the root
| File | What it does | Why it is there |
|---|---|---|
| `run_all.py` | The whole experiment: loads the data, builds the 5 subject-wise folds, runs the SVM (with grid search) and the GRU, computes the metrics and writes `results/` | One command reproduces every number in my CV |
| `requirements.txt` | The Python packages needed | So someone else can install the same things |
| `.gitignore` | Keeps the dataset, the cache and big files out of git | The dataset weighs 5 GB and its licence restricts redistribution |
| `README.md` | Public description, protocol, results and limitations | It is what a visitor sees on GitHub |
| `GUIDE.md` | This file | Your map for understanding the project |

### In `dhg/`
| File | What it does | Why it is there |
|---|---|---|
| `constants.py` | The 22 joint names, the 25 bones of the hand, the 3 subgraphs (long fingers, short fingers, palm), the 14 gesture names, the 30 frames per sequence | One place for the facts every other file needs |
| `data.py` | Reads each `skeleton_world.txt` (one frame = 66 numbers), fixes missing values, resamples to 30 frames, divides by hand size, prints a data-quality check | The models need inputs of the same size, whatever the hand or the speed. The cache means I don't re-read 5 GB every time |
| `features.py` | `graph_features` (465): for each subgraph and frame, spread of the joints, distance to the centroid, dispersion, plus speed, acceleration and palm displacement. `pose_features` (443): joint positions relative to the palm, averaged over 5 time segments, motion energy, palm trajectory, per-third displacement, path length. Together: 908 features | An SVM can't read a sequence, so I summarize it in numbers. The pose and trajectory features bring back the direction and orientation that the first descriptors lost |
| `models.py` | The GRU: BatchNorm, a linear layer down to 128, a 2-layer bidirectional GRU, mean and max pooling over time, dropout, then 14 outputs | The sequence model: it reads the gesture in order and learns its own features |
| `train_dl.py` | Builds the 9 channels per joint, the augmentation, the training loop, the best-epoch choice on 2 held-out training subjects, and the predictions on test subjects | Everything needed to train the GRU without ever touching the test subjects |
| `evaluate.py` | Accuracy and macro-F1 per fold, the summary table, the gain over the SVM, recall per gesture, confusion matrices, the comparison chart, `summary.md` | Honest reporting: every sequence is tested once, with mean and standard deviation |
| `__init__.py` | Makes `dhg/` a Python package and holds the version | It is what lets you write `from dhg.data import ...`, and it is why you must run from the project folder |

### In `results/`
| File | What it contains | Why it matters |
|---|---|---|
| `summary.md`, `summary.csv` | The final table: accuracy, macro-F1, time, gain over the SVM | The numbers I put in the CV |
| `per_fold.csv` | Each model's score on each fold | Shows the GRU wins on 5 folds out of 5, and how much the scores vary |
| `per_class_recall.csv` | Recall for each of the 14 gestures | Shows which gestures are still confused |
| `confusion_*.png`, `comparison.png` | Figures | For the report and the interview |
| `folds.csv` | Which sequence belongs to which fold | Lets anyone reproduce the same splits |
| `oof_predictions.npz` | The prediction for every sequence when it was in the test fold | You can recompute any metric without retraining |
| `run_info.json` | The settings, the data-quality check and the best epoch per fold | Traceability |

## 5. Answers for the interview, in my own words
- **Why a GRU?** A gesture is a sequence over time. A GRU reads the frames in order and keeps a memory of what came before, so I don't have to design the features by hand. I chose it over an LSTM because it has fewer parameters (2 gates instead of 3) and my sequences are short.
- **Why did the SVM go from 32.9% to 72.7%?** The first descriptors only measured distances, which lose direction and orientation. I saw in the confusion matrix that swipes in different directions were mixed up, so I added the pose and trajectory information.
- **How do I know the result isn't inflated?** The test people never appear in training, and the scaler, the feature selection and the choice of the best epoch never see them either.
- **What are the limits?** Five folds of four people, one seed, no ablation study, and not the official DHG split. I only evaluate the 14-gesture setting.
