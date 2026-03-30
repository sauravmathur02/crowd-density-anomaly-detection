from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier.weapon_classifier import WeaponClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 gun-vs-knife classifier on cropped weapon ROIs.")
    parser.add_argument(
        "--data-root",
        default=r"c:\Repo\Crowd and Anomaly Detection\project\classifier_data",
        help="Root directory containing train/ and val/ class folders.",
    )
    parser.add_argument(
        "--output",
        default=r"c:\Repo\Crowd and Anomaly Detection\project\models\classifier.pth",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu or cuda:0.")
    return parser.parse_args()


def build_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def build_dataloaders(data_root: Path, image_size: int, batch_size: int, workers: int):
    train_transform, val_transform = build_transforms(image_size)

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected classifier crop folders at data_root/train and data_root/val.")

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(images.size(0))

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
    )

    model = WeaponClassifier._build_model(num_classes=len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val_acc = 0.0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += int((preds == targets).sum().item())
            running_total += int(images.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_record)
        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "labels": list(train_dataset.classes),
                "input_size": int(args.image_size),
                "best_val_acc": best_val_acc,
            }
            torch.save(checkpoint, str(output_path))
            print(f"  -> saved best checkpoint to {output_path}")

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_payload = {
        "best_val_acc": best_val_acc,
        "classes": list(train_dataset.classes),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
