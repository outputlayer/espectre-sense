#!/usr/bin/env python3
"""Prepare WiFi CSI data for deep learning training.

Parses JSONL recordings, extracts per-node amplitude arrays,
computes baseline from empty room, creates sliding windows.
"""
import json
import numpy as np
import os
import sys
from pathlib import Path

# Config
NUM_NODES = 3
NUM_SUBCARRIERS = 56
SUBSAMPLE_RATE = 5       # ~109fps -> ~22Hz
WINDOW_SIZE = 100         # 100 frames at ~22Hz ≈ 4.5 sec
WINDOW_STRIDE = 20        # ~0.9 sec stride
TRIM_LYING_SECS = 240     # trim last 4 min of lying data

CLASSES = {
    "empty": 0,
    "lying": 1,
    "walking": 2,
    "sitting": 3,
}

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "prepared"


def parse_recording(filepath: str, max_frames: int = None) -> np.ndarray:
    """Parse JSONL recording into (num_frames, NUM_NODES, NUM_SUBCARRIERS) array."""
    node_latest = {i: np.zeros(NUM_SUBCARRIERS) for i in range(1, NUM_NODES + 1)}
    frames = []
    count = 0

    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Update latest amplitudes for each node in this tick
            for node in data.get("nodes", []):
                nid = node.get("node_id")
                if nid and 1 <= nid <= NUM_NODES:
                    amp = node.get("amplitude", [])
                    arr = np.zeros(NUM_SUBCARRIERS)
                    arr[:min(NUM_SUBCARRIERS, len(amp))] = amp[:NUM_SUBCARRIERS]
                    node_latest[nid] = arr

            # Stack all 3 nodes into one frame
            frame = np.stack([node_latest[i] for i in range(1, NUM_NODES + 1)])
            frames.append(frame)
            count += 1

            if max_frames and count >= max_frames:
                break

    arr = np.array(frames, dtype=np.float32)
    print(f"  Parsed {filepath}: {arr.shape[0]} frames")
    return arr


def subsample(data: np.ndarray, rate: int) -> np.ndarray:
    """Take every `rate`-th frame."""
    return data[::rate]


def compute_baseline(empty_data: np.ndarray) -> np.ndarray:
    """Compute per-node, per-subcarrier mean from empty room data.
    Returns shape (NUM_NODES, NUM_SUBCARRIERS)."""
    return empty_data.mean(axis=0)


def create_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create sliding windows from (num_frames, nodes, subcarriers).
    Returns (num_windows, window_size, nodes * subcarriers)."""
    n_frames = data.shape[0]
    # Flatten nodes and subcarriers
    flat = data.reshape(n_frames, -1)  # (frames, nodes*subcarriers)

    windows = []
    for start in range(0, n_frames - window_size + 1, stride):
        windows.append(flat[start:start + window_size])

    return np.array(windows, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    recordings = {
        "empty": DATA_DIR / "train_empty_v3.jsonl",
        "lying": DATA_DIR / "train_lying_v3.jsonl",
        "walking": DATA_DIR / "train_walking_v3.jsonl",
        "sitting": DATA_DIR / "train_sitting_v3.jsonl",
    }

    # Check all files exist
    for name, path in recordings.items():
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # 1. Parse all recordings
    print("=== Parsing recordings ===")
    raw_data = {}
    for name, path in recordings.items():
        raw_data[name] = parse_recording(str(path))

    # 2. Trim lying data (last 4 min = last ~26000 frames at ~109fps)
    lying_trim = int(TRIM_LYING_SECS * 109)  # approximate fps
    if raw_data["lying"].shape[0] > lying_trim:
        original = raw_data["lying"].shape[0]
        raw_data["lying"] = raw_data["lying"][:-lying_trim]
        print(f"  Trimmed lying: {original} -> {raw_data['lying'].shape[0]} frames")

    # 3. Subsample
    print(f"\n=== Subsampling (1/{SUBSAMPLE_RATE}) ===")
    for name in raw_data:
        original = raw_data[name].shape[0]
        raw_data[name] = subsample(raw_data[name], SUBSAMPLE_RATE)
        print(f"  {name}: {original} -> {raw_data[name].shape[0]} frames")

    # 4. Compute baseline from empty room
    print("\n=== Computing baseline ===")
    baseline = compute_baseline(raw_data["empty"])
    print(f"  Baseline shape: {baseline.shape}")
    means = [baseline[i].mean() for i in range(3)]; print(f"  Baseline mean per node: {means}")
    np.save(OUTPUT_DIR / "baseline.npy", baseline)

    # 5. Subtract baseline
    print("\n=== Subtracting baseline ===")
    for name in raw_data:
        raw_data[name] = raw_data[name] - baseline[np.newaxis, :, :]
        print(f"  {name}: mean abs after baseline sub: {np.abs(raw_data[name]).mean():.4f}")

    # 6. Create sliding windows
    print(f"\n=== Creating windows (size={WINDOW_SIZE}, stride={WINDOW_STRIDE}) ===")
    all_X = []
    all_y = []

    for name, label in CLASSES.items():
        windows = create_windows(raw_data[name], WINDOW_SIZE, WINDOW_STRIDE)
        print(f"  {name} (label={label}): {windows.shape[0]} windows, shape={windows.shape}")
        all_X.append(windows)
        all_y.append(np.full(windows.shape[0], label, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\n  Total: {X.shape[0]} windows, input shape: {X.shape[1:]}")

    # 7. Shuffle and split
    print("\n=== Train/Val split ===")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    val_ratio = 0.2
    val_size = int(len(X) * val_ratio)
    X_val, y_val = X[:val_size], y[:val_size]
    X_train, y_train = X[val_size:], y[val_size:]

    print(f"  Train: {X_train.shape[0]} windows")
    print(f"  Val:   {X_val.shape[0]} windows")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Class distribution (val):   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # 8. Normalize — compute per-feature mean/std from training set
    print("\n=== Normalizing ===")
    # Reshape to (samples * timesteps, features) for stats
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    feat_mean = flat_train.mean(axis=0)
    feat_std = flat_train.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0  # avoid division by zero

    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    # 9. Save
    print("\n=== Saving ===")
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
    np.save(OUTPUT_DIR / "feat_std.npy", feat_std)

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"  X_train: {X_train.shape} ({X_train.nbytes / 1e6:.1f} MB)")
    print(f"  X_val:   {X_val.shape} ({X_val.nbytes / 1e6:.1f} MB)")

    # Summary
    print("\n=== Summary ===")
    print(f"  Classes: {list(CLASSES.keys())}")
    print(f"  Window: {WINDOW_SIZE} frames @ ~22Hz = ~{WINDOW_SIZE/22:.1f}s")
    print(f"  Features per frame: {NUM_NODES * NUM_SUBCARRIERS}")
    print(f"  Input tensor: ({WINDOW_SIZE}, {NUM_NODES * NUM_SUBCARRIERS})")


if __name__ == "__main__":
    main()
