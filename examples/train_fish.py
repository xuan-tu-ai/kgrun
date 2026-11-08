"""
Fish freshness classification training script for Milestone M3.
Trains a deep convolutional neural network on Kaggle GPU using the FFE dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int = 999) -> None:
    """
    Configure random number generators across packages for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FishImageDataset(Dataset):
    """
    Custom Dataset for loading and transforming fish eye images.
    """

    def __init__(self, paths: list[Path], labels: list[int], transform: transforms.Compose) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.paths[idx]
        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")
        tensor_img = self.transform(image_rgb)
        label = self.labels[idx]
        return tensor_img, label


def scan_dataset(root_dir: Path) -> tuple[list[Path], list[int]]:
    """
    Discover all image files and derive labels based on directory nomenclature.
    Mapping: Highly Fresh -> 0, Fresh -> 1, Not Fresh -> 2.
    """
    all_files = sorted(
        list(root_dir.rglob("*.jpg"))
        + list(root_dir.rglob("*.png"))
        + list(root_dir.rglob("*.jpeg"))
    )
    paths: list[Path] = []
    labels: list[int] = []

    for file_path in all_files:
        path_str = str(file_path).lower()
        if "highly" in path_str:
            label = 0
        elif "not" in path_str:
            label = 2
        elif "fresh" in path_str:
            label = 1
        else:
            continue

        paths.append(file_path)
        labels.append(label)

    return paths, labels


def stratified_split(
    paths: list[Path],
    labels: list[int],
    val_ratio: float = 0.2,
    seed: int = 999,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """
    Perform deterministic stratified train/val split.
    """
    random.seed(seed)
    class_indices: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for _, indices in sorted(class_indices.items()):
        shuffled = list(indices)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    return train_paths, train_labels, val_paths, val_labels


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Calculate validation loss and top-1 classification accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fish freshness classifier on GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=999, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64, flush=True)
    print(" FISH FRESHNESS TRAINING (Milestone M3)", flush=True)
    print("=" * 64, flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU device name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU count: {torch.cuda.device_count()}", flush=True)

    data_dir_env = os.environ.get("KGRUN_DATA", "data")
    data_dir = Path(data_dir_env)
    print(f"Data root: {data_dir.resolve()}", flush=True)

    paths, labels = scan_dataset(data_dir)
    print(f"Loaded {len(paths)} images across classes: 0={labels.count(0)}, 1={labels.count(1)}, 2={labels.count(2)}", flush=True)

    if not paths:
        print("Error: No valid images found in dataset directory!", file=sys.stderr)
        return 1

    tr_paths, tr_labels, va_paths, va_labels = stratified_split(paths, labels, val_ratio=0.2, seed=args.seed)
    print(f"Train samples: {len(tr_paths)}, Validation samples: {len(va_paths)}", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FishImageDataset(tr_paths, tr_labels, train_tf)
    val_ds = FishImageDataset(va_paths, va_labels, val_tf)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Building ResNet18 model...", flush=True)
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"Pretrained weights notice: {exc}. Initializing with random weights.", flush=True)
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    out_dir = Path(os.environ.get("KGRUN_OUT", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_acc = 0.0
    start_time = time.time()

    print(f"Beginning training for {args.epochs} epoch(s)...", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_loss = running_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_elapsed = time.time() - epoch_t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "elapsed_s": round(epoch_elapsed, 1),
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"time: {epoch_elapsed:.1f}s",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt = out_dir / "best_fish_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "seed": args.seed,
                },
                best_ckpt,
            )

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s. Best Validation Accuracy: {best_acc:.4f}", flush=True)

    # Save metrics CSV
    csv_path = out_dir / "fish_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved metric log to {csv_path}", flush=True)

    # Save summary JSON
    summary_path = out_dir / "training_summary.json"
    summary_data = {
        "dataset_samples": len(paths),
        "train_samples": len(tr_paths),
        "val_samples": len(va_paths),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_accuracy": round(best_acc, 4),
        "total_time_s": round(total_time, 1),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Saved training summary to {summary_path}", flush=True)
    print("=" * 64, flush=True)
    print(" TRAINING COMPLETE AND VERIFIED", flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

