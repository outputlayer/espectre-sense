#!/usr/bin/env python3
"""Train CNN-LSTM model on WiFi CSI data for room presence classification."""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from csi_model import CSINet, CSINetLight, count_params

PREPARED_DIR = Path(__file__).parent / "prepared"
MODEL_DIR = Path(__file__).parent / "models"
CLASSES = ["empty", "lying", "walking", "sitting"]

# Training config
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10  # early stopping


def load_data():
    print("Loading prepared data...")
    X_train = np.load(PREPARED_DIR / "X_train.npy")
    y_train = np.load(PREPARED_DIR / "y_train.npy")
    X_val = np.load(PREPARED_DIR / "X_val.npy")
    y_val = np.load(PREPARED_DIR / "y_val.npy")

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Balance training data via oversampling minority classes
    class_counts = np.bincount(y_train)
    max_count = class_counts.max()
    balanced_idx = []
    for c in range(len(class_counts)):
        cls_idx = np.where(y_train == c)[0]
        if len(cls_idx) < max_count:
            extra = np.random.choice(cls_idx, max_count - len(cls_idx), replace=True)
            balanced_idx.extend(cls_idx)
            balanced_idx.extend(extra)
        else:
            balanced_idx.extend(cls_idx)
    np.random.shuffle(balanced_idx)
    X_train = X_train[balanced_idx]
    y_train = y_train[balanced_idx]
    print(f"  Balanced train: {X_train.shape}")
    print(f"  Balanced classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # To tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=False)

    return train_loader, val_loader, X_train.shape[-1]


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += X_batch.size(0)
            all_preds.extend(pred.numpy())
            all_labels.extend(y_batch.numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def confusion_matrix_report(preds, labels, class_names):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    print("\nConfusion Matrix:")
    header = "          " + " ".join(f"{c:>8s}" for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:>8s}: " + " ".join(f"{cm[i][j]:>8d}" for j in range(n))
        print(row)

    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:>8s}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")


def export_model(model, input_features, model_name):
    """Export model weights as JSON for Rust deployment."""
    MODEL_DIR.mkdir(exist_ok=True)

    # Save PyTorch model
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}.pt")


    # Save metadata
    meta = {
        "model": model_name,
        "classes": CLASSES,
        "input_shape": [100, input_features],
        "params": count_params(model),
    }
    with open(MODEL_DIR / f"{model_name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExported: {MODEL_DIR / model_name}.pt, .onnx, _meta.json")


def train_model(model_class, model_name, input_features, train_loader, val_loader):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model = model_class(input_features=input_features, num_classes=len(CLASSES))
    print(f"Parameters: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
    )

    best_val_acc = 0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"LR={lr:.6f} | {elapsed:.1f}s")
        sys.stdout.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # Save best
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Load best and evaluate
    model.load_state_dict(torch.load(MODEL_DIR / f"{model_name}_best.pt", weights_only=True))
    val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)

    print(f"\nBest epoch: {best_epoch}, Val accuracy: {val_acc:.3f}")
    confusion_matrix_report(preds, labels, CLASSES)

    # Export
    export_model(model, input_features, model_name)

    return model, best_val_acc


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    train_loader, val_loader, input_features = load_data()

    results = {}

    # Train CNN-LSTM (full model)
    model1, acc1 = train_model(CSINet, "csi_cnn_lstm", input_features, train_loader, val_loader)
    results["CSINet"] = acc1

    # Train lightweight CNN-only
    model2, acc2 = train_model(CSINetLight, "csi_cnn_light", input_features, train_loader, val_loader)
    results["CSINetLight"] = acc2

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:20s}: {acc*100:.1f}%")

    best = max(results, key=results.get)
    print(f"\n  Best model: {best} ({results[best]*100:.1f}%)")


if __name__ == "__main__":
    main()
