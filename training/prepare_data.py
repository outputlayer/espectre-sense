#!/usr/bin/env python3
"""Prepare WiFi CSI data — v4: fresh data only (same-day, same conditions).

Uses only v4/v5/v6 recordings from today's session.
Ensures zero distribution shift between train and val.
Old data excluded to eliminate cross-session contamination.
"""
import json
import numpy as np
import sys
from pathlib import Path

NUM_NODES = 3
NUM_SUBCARRIERS = 56
NUM_FEATURES = NUM_NODES * NUM_SUBCARRIERS
SUBSAMPLE_RATE = 5
WINDOW_SIZE = 40
WINDOW_STRIDE = 10
VAL_RATIO = 0.3

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
    print(f"  Parsed {filepath.name}: {arr.shape[0]} frames ({arr.shape[0]/109:.0f}s)")
    return arr


def l2_normalize(data):
    n = data.shape[0]
    flat = data.reshape(n, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    flat = flat / norms
    return flat.reshape(n, NUM_NODES, NUM_SUBCARRIERS)


def create_windows(data, window_size, stride):
    n = data.shape[0]
    flat = data.reshape(n, -1)
    windows = []
    for start in range(0, n - window_size + 1, stride):
        windows.append(flat[start:start + window_size])
    if windows:
        return np.array(windows, dtype=np.float32)
    return np.zeros((0, window_size, flat.shape[1]), dtype=np.float32)


def augment_window(window, n_augments=3):
    """More augmentation to compensate for smaller dataset."""
    augmented = []
    for _ in range(n_augments):
        w = window.copy()
        w += np.random.normal(0, 0.02, w.shape).astype(np.float32)
        # Drift augmentation
        drift = np.random.normal(0, 0.15, (1, w.shape[1])).astype(np.float32)
        w += drift
        scale = np.random.uniform(0.85, 1.15, (1, w.shape[1])).astype(np.float32)
        w *= scale
        augmented.append(w)
    return augmented


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # FRESH DATA ONLY — same-day recordings
    recordings = {
        "empty":   [DATA_DIR / "train_empty_v4.jsonl"],
        "lying":   [DATA_DIR / "train_lying_v6.jsonl"],
        "walking": [DATA_DIR / "train_walking_v5.jsonl"],
        "sitting": [DATA_DIR / "train_sitting_v5.jsonl"],
    }

    for name, paths in recordings.items():
        for path in paths:
            if not path.exists():
                print(f"ERROR: {path} not found"); sys.exit(1)

    # 1. Parse
    print("=== Parsing fresh recordings ===")
    raw_data = {}
    for name, paths in recordings.items():
        parts = [parse_recording(p) for p in paths]
        raw_data[name] = np.concatenate(parts, axis=0)

    # 2. L2 normalize
    print("\n=== L2 normalization ===")
    for name in raw_data:
        raw_data[name] = l2_normalize(raw_data[name])

    # 3. Baseline from empty
    baseline = raw_data["empty"].mean(axis=0)
    np.save(OUTPUT_DIR / "baseline.npy", baseline)
    print(f"  Baseline shape: {baseline.shape}, mean: {baseline.mean():.6f}")

    # 4. Subtract baseline
    for name in raw_data:
        raw_data[name] = raw_data[name] - baseline[np.newaxis, :, :]

    # 5. Subsample
    print(f"\n=== Subsample (1/{SUBSAMPLE_RATE}) ===")
    for name in raw_data:
        orig = raw_data[name].shape[0]
        raw_data[name] = raw_data[name][::SUBSAMPLE_RATE]
        print(f"  {name}: {orig} -> {raw_data[name].shape[0]} frames")

    # 6. Temporal split + windows + augmentation
    print(f"\n=== Temporal split (70/30) ===")
    train_X, train_y = [], []
    val_X, val_y = [], []

    for name, label in CLASSES.items():
        data = raw_data[name]
        n = data.shape[0]
        split = int(n * (1 - VAL_RATIO))

        tw = create_windows(data[:split], WINDOW_SIZE, WINDOW_STRIDE)
        vw = create_windows(data[split:], WINDOW_SIZE, WINDOW_STRIDE)

        # More augmentation for smaller dataset (3x instead of 2x)
        aug_windows = []
        for i in range(tw.shape[0]):
            aug_windows.extend(augment_window(tw[i], n_augments=3))
        if aug_windows:
            tw = np.concatenate([tw, np.array(aug_windows, dtype=np.float32)], axis=0)

        print(f"  {name}: train={tw.shape[0]} (incl. 3x aug), val={vw.shape[0]}")
        train_X.append(tw)
        train_y.append(np.full(tw.shape[0], label, dtype=np.int64))
        val_X.append(vw)
        val_y.append(np.full(vw.shape[0], label, dtype=np.int64))

    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)

    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\n  Total: train={X_train.shape[0]}, val={X_val.shape[0]}")
    print(f"  Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # 7. Global normalization
    print("\n=== Global normalization ===")
    flat = X_train.reshape(-1, X_train.shape[-1])
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
    np.save(OUTPUT_DIR / "feat_std.npy", feat_std)

    # 8. Save
    print("\n=== Saving ===")
    for arr_name, arr in [("X_train", X_train), ("y_train", y_train),
                           ("X_val", X_val), ("y_val", y_val)]:
        np.save(OUTPUT_DIR / f"{arr_name}.npy", arr)

    print(f"  X_train: {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")
    print(f"  X_val:   {X_val.shape} ({X_val.nbytes/1e6:.1f} MB)")

    print(f"\n=== Pipeline: FRESH ONLY ===")
    print(f"  L2 norm → baseline (v4 empty) → global norm → drift augmentation")
    print(f"  Zero cross-session contamination")


if __name__ == "__main__":
    main()
