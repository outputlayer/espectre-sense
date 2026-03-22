#!/usr/bin/env python3
"""Prepare WiFi CSI data for deep learning training.

v2: Temporal train/val split (no leakage), smaller window, augmentation.
"""
import json
import numpy as np
import sys
from pathlib import Path

NUM_NODES = 3
NUM_SUBCARRIERS = 56
SUBSAMPLE_RATE = 5
WINDOW_SIZE = 40          # ~40 frames at ~22Hz ≈ 1.8 sec
WINDOW_STRIDE = 10        # ~0.45 sec stride
TRIM_LYING_V3_SECS = 240   # trim bad end of v3 lying
VAL_RATIO = 0.3           # last 30% of each recording → val

CLASSES = {"empty": 0, "lying": 1, "walking": 2, "sitting": 3}
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "prepared"


def parse_recording(filepath):
    node_latest = {i: np.zeros(NUM_SUBCARRIERS) for i in range(1, NUM_NODES + 1)}
    frames = []
    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for node in data.get("nodes", []):
                nid = node.get("node_id")
                if nid and 1 <= nid <= NUM_NODES:
                    amp = node.get("amplitude", [])
                    arr = np.zeros(NUM_SUBCARRIERS)
                    arr[:min(NUM_SUBCARRIERS, len(amp))] = amp[:NUM_SUBCARRIERS]
                    node_latest[nid] = arr
            frame = np.stack([node_latest[i] for i in range(1, NUM_NODES + 1)])
            frames.append(frame)
    arr = np.array(frames, dtype=np.float32)
    print(f"  Parsed {filepath}: {arr.shape[0]} frames")
    return arr


def create_windows(data, window_size, stride):
    n = data.shape[0]
    flat = data.reshape(n, -1)
    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(flat[start:start + window_size])
    return np.array(windows, dtype=np.float32) if windows else np.zeros((0, window_size, flat.shape[1]), dtype=np.float32)


def augment_window(window, n_augments=2):
    """Generate augmented copies: noise + amplitude scaling."""
    augmented = []
    for _ in range(n_augments):
        w = window.copy()
        # Gaussian noise (small)
        w += np.random.normal(0, 0.02, w.shape).astype(np.float32)
        # Random amplitude scaling per feature (0.9-1.1)
        scale = np.random.uniform(0.9, 1.1, (1, w.shape[1])).astype(np.float32)
        w *= scale
        augmented.append(w)
    return augmented


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    recordings = {
        "empty": [DATA_DIR / "train_empty_v3.jsonl"],
        "lying": [DATA_DIR / "train_lying_v3.jsonl", DATA_DIR / "train_lying_v4.jsonl"],
        "walking": [DATA_DIR / "train_walking_v3.jsonl", DATA_DIR / "train_walking_v4.jsonl"],
        "sitting": [DATA_DIR / "train_sitting_v3.jsonl", DATA_DIR / "train_sitting_v4.jsonl"],
    }

    # Per-file trimming (seconds from end)
    trim_map = {
        str(DATA_DIR / "train_lying_v3.jsonl"): int(TRIM_LYING_V3_SECS * 109),
    }

    for name, paths in recordings.items():
        for path in paths:
            if not path.exists():
                print(f"ERROR: {path} not found"); sys.exit(1)

    # 1. Parse
    print("=== Parsing recordings ===")
    raw_data = {}
    for name, paths in recordings.items():
        parts = []
        for p in paths:
            part = parse_recording(str(p))
            trim = trim_map.get(str(p), 0)
            if trim > 0 and part.shape[0] > trim:
                orig = part.shape[0]
                part = part[:-trim]
                print(f"    Trimmed {p.name}: {orig} -> {part.shape[0]}")
            parts.append(part)
        raw_data[name] = np.concatenate(parts, axis=0)
        if len(parts) > 1:
            print(f"  {name}: merged {len(parts)} files -> {raw_data[name].shape[0]} frames")


    # 3. Subsample
    print(f"\n=== Subsampling (1/{SUBSAMPLE_RATE}) ===")
    for name in raw_data:
        orig = raw_data[name].shape[0]
        raw_data[name] = raw_data[name][::SUBSAMPLE_RATE]
        print(f"  {name}: {orig} -> {raw_data[name].shape[0]} frames")

    # 4. Baseline from empty
    print("\n=== Computing baseline ===")
    baseline = raw_data["empty"].mean(axis=0)
    np.save(OUTPUT_DIR / "baseline.npy", baseline)
    print(f"  Baseline shape: {baseline.shape}")

    # 5. Subtract baseline
    for name in raw_data:
        raw_data[name] = raw_data[name] - baseline[np.newaxis, :, :]

    # 6. TEMPORAL SPLIT: first 70% → train, last 30% → val (per recording)
    print(f"\n=== Temporal split (train={1-VAL_RATIO:.0%} / val={VAL_RATIO:.0%}) ===")
    train_X, train_y = [], []
    val_X, val_y = [], []

    for name, label in CLASSES.items():
        data = raw_data[name]
        n = data.shape[0]
        split = int(n * (1 - VAL_RATIO))

        train_part = data[:split]
        val_part = data[split:]

        tw = create_windows(train_part, WINDOW_SIZE, WINDOW_STRIDE)
        vw = create_windows(val_part, WINDOW_SIZE, WINDOW_STRIDE)

        # Augment train data
        aug_windows = []
        for i in range(tw.shape[0]):
            aug_windows.extend(augment_window(tw[i], n_augments=2))
        if aug_windows:
            aug_arr = np.array(aug_windows, dtype=np.float32)
            tw = np.concatenate([tw, aug_arr], axis=0)

        print(f"  {name}: train={tw.shape[0]} windows (incl. aug), val={vw.shape[0]} windows")

        train_X.append(tw)
        train_y.append(np.full(tw.shape[0], label, dtype=np.int64))
        val_X.append(vw)
        val_y.append(np.full(vw.shape[0], label, dtype=np.int64))

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)

    # Shuffle train
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\n  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"  Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val classes:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # 7. Normalize
    print("\n=== Normalizing ===")
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    feat_mean = flat_train.mean(axis=0)
    feat_std = flat_train.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # 8. Save
    print("\n=== Saving ===")
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                       ("X_val", X_val), ("y_val", y_val),
                       ("feat_mean", feat_mean), ("feat_std", feat_std)]:
        np.save(OUTPUT_DIR / f"{name}.npy", arr)

    print(f"  X_train: {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")
    print(f"  X_val:   {X_val.shape} ({X_val.nbytes/1e6:.1f} MB)")

    print(f"\n=== Summary ===")
    print(f"  Window: {WINDOW_SIZE} frames @ ~{109/SUBSAMPLE_RATE:.0f}Hz = ~{WINDOW_SIZE/(109/SUBSAMPLE_RATE):.1f}s")
    print(f"  Features: {NUM_NODES * NUM_SUBCARRIERS}")
    print(f"  Temporal split: NO LEAKAGE")


if __name__ == "__main__":
    main()
