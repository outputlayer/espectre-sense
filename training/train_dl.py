#!/usr/bin/env python3
"""Train CSINetLight on WiFi CSI data and export weights for Rust inference."""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from csi_model import CSINetLight, count_params

PREPARED_DIR = Path(__file__).parent / "prepared"
MODEL_DIR = Path(__file__).parent / "models"
SERVER_MODEL_DIR = Path(__file__).parent.parent / "server" / "models"
CLASSES = ["empty", "lying", "walking", "sitting"]

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
PATIENCE = 15


def load_data():
    print("Loading prepared data...")
    X_train = np.load(PREPARED_DIR / "X_train.npy")
    y_train = np.load(PREPARED_DIR / "y_train.npy")
    X_val = np.load(PREPARED_DIR / "X_val.npy")
    y_val = np.load(PREPARED_DIR / "y_val.npy")

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val classes:   {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # Balance train via oversampling
    class_counts = np.bincount(y_train)
    max_count = class_counts.max()
    balanced_idx = []
    for c in range(len(class_counts)):
        cls_idx = np.where(y_train == c)[0]
        extra = np.random.choice(cls_idx, max_count - len(cls_idx), replace=True) if len(cls_idx) < max_count else np.array([], dtype=int)
        balanced_idx.extend(cls_idx)
        balanced_idx.extend(extra)
    np.random.shuffle(balanced_idx)
    X_train, y_train = X_train[balanced_idx], y_train[balanced_idx]
    print(f"  Balanced: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2),
            X_train.shape[-1])


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X, y in loader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return loss_sum / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            loss = criterion(out, y)
            loss_sum += loss.item() * X.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += X.size(0)
            all_preds.extend(pred.numpy())
            all_labels.extend(y.numpy())
    return loss_sum / total, correct / total, np.array(all_preds), np.array(all_labels)


def confusion_matrix_report(preds, labels):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    print("\nConfusion Matrix:")
    print("          " + " ".join(f"{c:>8s}" for c in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"{name:>8s}: " + " ".join(f"{cm[i][j]:>8d}" for j in range(n)))
    for i, name in enumerate(CLASSES):
        tp = cm[i][i]
        fp, fn = cm[:, i].sum() - tp, cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {name:>8s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")


def export_weights_json(model, path):
    """Export model weights as JSON for pure-Rust inference."""
    state = model.state_dict()
    weights = {}
    for key, tensor in state.items():
        weights[key] = tensor.cpu().numpy().flatten().tolist()
    data = {"classes": CLASSES, "weights": weights}
    with open(path, "w") as f:
        json.dump(data, f)
    size_mb = path.stat().st_size / 1e6
    print(f"  Exported weights: {path} ({size_mb:.1f} MB)")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    train_loader, val_loader, input_features = load_data()

    model = CSINetLight(input_features=input_features, num_classes=len(CLASSES))
    print(f"\nCSINetLight: {count_params(model):,} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    MODEL_DIR.mkdir(exist_ok=True)
    best_val_acc, best_epoch, no_improve = 0, 0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, preds, labels = validate(model, val_loader, criterion)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {tl:.4f} {ta:.3f} | Val: {vl:.4f} {va:.3f} | LR={lr:.6f} | {time.time()-t0:.1f}s")
        sys.stdout.flush()

        if va > best_val_acc:
            best_val_acc, best_epoch, no_improve = va, epoch, 0
            torch.save(model.state_dict(), MODEL_DIR / "csi_cnn_light_best.pt")
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Load best and final eval
    model.load_state_dict(torch.load(MODEL_DIR / "csi_cnn_light_best.pt", weights_only=True))
    _, va, preds, labels = validate(model, val_loader, criterion)
    print(f"\nBest epoch: {best_epoch}, Val accuracy: {va:.3f}")
    confusion_matrix_report(preds, labels)

    # Export
    torch.save(model.state_dict(), MODEL_DIR / "csi_cnn_light.pt")
    export_weights_json(model, MODEL_DIR / "csi_light_weights.json")

    # Copy to server/models
    SERVER_MODEL_DIR.mkdir(exist_ok=True)
    import shutil
    for f in ["csi_light_weights.json"]:
        shutil.copy2(MODEL_DIR / f, SERVER_MODEL_DIR / f)
    # Copy normalization files
    for f in ["baseline.npy", "feat_mean.npy", "feat_std.npy"]:
        shutil.copy2(PREPARED_DIR / f, SERVER_MODEL_DIR / f)
    print(f"\n  Copied model + norms to {SERVER_MODEL_DIR}/")


if __name__ == "__main__":
    main()
