"""Loading, resampling and normalisation of DHG-2016 skeleton sequences.

Pipeline in 3 steps (each one is a function below):
    1. load_dataset : read every skeleton_world.txt and give all sequences the same length (30 frames)
    2. normalize    : make the sequences independent of hand size
    3. quality_report : print a quick sanity check of the data

Shapes used everywhere:  N = number of sequences, T = frames, 22 = joints, 3 = (x, y, z).
"""

import re
from pathlib import Path

import numpy as np

from .constants import NUM_JOINTS, TARGET_FRAMES


# ----------------------------------------------------------------------------
# Small helpers for reading files
# ----------------------------------------------------------------------------
def _natural_key(path: Path):
    """Sort key so that 'subject_2' comes before 'subject_10' (plain sorting would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _subfolders(folder: Path):
    """Sub-folders of `folder`, in natural order (1, 2, ..., 10 instead of 1, 10, 2)."""
    return sorted((p for p in folder.iterdir() if p.is_dir()), key=_natural_key)


def _number_in_name(folder: Path) -> int:
    """'subject_12' -> 12."""
    return int(folder.name.split("_")[-1])


def _read_skeleton(path: Path) -> np.ndarray:
    """Read one skeleton_world.txt -> array (T, 22, 3).

    Each line of the file is one frame: 66 numbers = 22 joints x (x, y, z).
    Lines that do not have exactly 66 numbers are ignored.
    """
    frames = []
    with open(path) as f:
        for line in f:
            values = line.split()
            if len(values) == NUM_JOINTS * 3:
                frames.append(np.asarray(values, dtype=np.float64))
    if not frames:
        return np.empty((0, NUM_JOINTS, 3))          # empty file -> caller skips it
    return np.asarray(frames).reshape(-1, NUM_JOINTS, 3)


def resample_linear(seq: np.ndarray, target: int) -> np.ndarray:
    """Give a sequence exactly `target` frames: (T, 22, 3) -> (target, 22, 3).

    Sequences have different lengths (a fast swipe has fewer frames than a slow one).
    We pick `target` evenly spaced moments in time; when a moment falls between two
    real frames, we blend those two frames (linear interpolation).
    """
    n = len(seq)
    if n == 1:                                        # only one frame: just repeat it
        return np.repeat(seq, target, axis=0)
    pos = np.linspace(0, n - 1, target)               # wanted moments, e.g. 3.4, 4.1, ...
    before = np.floor(pos).astype(int)                # real frame just before each moment
    after = np.minimum(before + 1, n - 1)             # real frame just after
    w = (pos - before)[:, None, None]                 # how close we are to the "after" frame
    return seq[before] * (1.0 - w) + seq[after] * w


# ----------------------------------------------------------------------------
# Step 1: load the whole dataset
# ----------------------------------------------------------------------------
def load_dataset(root, target_frames: int = TARGET_FRAMES, cache=None, num_classes: int = 14):
    """Read DHG2016/gesture_X/finger_Y/subject_Z/essai_W/skeleton_world.txt for all folders.

    Returns a dict:
        X        (N, T, 22, 3) float32  joint positions, in the sensor's own unit
        y        (N,)  gesture label, 0..13
        subject  (N,)  subject id (used later to split train/test BY PERSON)
        finger   (N,)  1 or 2 (gesture performed with one finger or the whole hand)
        essai    (N,)  repetition number
        n_orig   (N,)  number of frames before resampling

    If `cache` is a file path, the result is saved there the first time and simply
    reloaded on later runs (so the 5 GB dataset is not needed again).
    """
    # --- reuse the cache when it exists ---
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    root = Path(root)
    sequences, labels, subjects, fingers, essais, lengths = [], [], [], [], [], []
    n_repaired = 0

    # --- walk the folder tree: gesture > finger > subject > essai ---
    for g in range(1, num_classes + 1):
        gesture_dir = root / f"gesture_{g}"
        if not gesture_dir.exists():
            continue
        for finger_dir in _subfolders(gesture_dir):
            for subject_dir in _subfolders(finger_dir):
                for essai_dir in _subfolders(subject_dir):
                    skeleton_file = essai_dir / "skeleton_world.txt"
                    if not skeleton_file.exists():
                        continue

                    seq = _read_skeleton(skeleton_file)
                    if len(seq) == 0:                         # nothing usable in the file
                        continue
                    if not np.isfinite(seq).all():            # NaN / inf values: replace by 0
                        n_repaired += 1
                        seq = np.nan_to_num(seq)

                    # same number of frames for every sequence, then store with its labels
                    sequences.append(resample_linear(seq, target_frames).astype(np.float32))
                    labels.append(g - 1)                      # gesture 1..14 -> class 0..13
                    subjects.append(_number_in_name(subject_dir))
                    fingers.append(_number_in_name(finger_dir))
                    essais.append(_number_in_name(essai_dir))
                    lengths.append(len(seq))

    if not sequences:
        raise FileNotFoundError(
            f"No skeleton_world.txt found under {root}. Expected "
            "gesture_X/finger_Y/subject_Z/essai_W/skeleton_world.txt"
        )

    data = dict(
        X=np.stack(sequences), y=np.array(labels), subject=np.array(subjects),
        finger=np.array(fingers), essai=np.array(essais), n_orig=np.array(lengths),
    )
    print(f"Loaded {len(sequences)} sequences ({n_repaired} with non-finite values repaired)")

    if cache is not None:                                     # save for next time
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
    return data


# ----------------------------------------------------------------------------
# Step 2: remove the effect of hand size
# ----------------------------------------------------------------------------
def normalize(X: np.ndarray):
    """Divide each sequence by the size of that hand, so big and small hands look alike.

    Hand size = median distance between wrist (joint 0) and palm centre (joint 1)
    over the sequence's frames. We use ONE size per sequence (not one per frame),
    because a per-frame size would add noise from frame to frame.
    Positions are not re-centred here: the hand's overall movement is kept, since
    it is what distinguishes swipes (centring is done later, in the features step).

    Returns (normalised X as float32, the hand size of each sequence).
    """
    wrist, palm = X[:, :, 0, :], X[:, :, 1, :]
    dist = np.linalg.norm(palm - wrist, axis=-1)              # (N, T) distance in every frame
    size = np.maximum(np.median(dist, axis=1), 1e-9)          # (N,) one size per sequence
    return (X / size[:, None, None, None]).astype(np.float32), size


# ----------------------------------------------------------------------------
# Step 3: sanity check
# ----------------------------------------------------------------------------
def quality_report(X: np.ndarray, s: np.ndarray) -> dict:
    """Print a short summary to spot problems: unit (m or mm), hand size, jumps between frames.

    `s` is the hand size returned by `normalize`.
    """
    # a hand is ~5 cm: a size below 1 means metres, otherwise millimetres
    unit = "metres" if np.median(s) < 1.0 else "millimetres"
    # biggest joint movement between two frames, in "palm lengths" (very large = glitch)
    step = np.linalg.norm(np.diff(X, axis=1), axis=-1).max(axis=-1)
    jump = step / s[:, None]
    report = dict(
        n_sequences=int(len(X)),
        n_frames_total=int(X.shape[0] * X.shape[1]),
        unit_guess=unit,
        median_wrist_palm=float(np.median(s)),
        p99_joint_speed_in_palm_lengths=float(np.percentile(jump, 99)),
    )
    print("Data quality:", report)
    return report
