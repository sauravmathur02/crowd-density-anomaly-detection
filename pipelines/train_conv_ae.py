from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Ensure repo root is on path when run as a script (e.g. subprocess from prepare_assets).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.autoencoder import ConvAE


class FramesDataset(Dataset):
    def __init__(self, frames_root: Path, max_images: Optional[int] = None) -> None:
        self.frames_root = frames_root
        self.paths: List[Path] = []
        for p in frames_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self.paths.append(p)
        self.paths.sort()

        if max_images is not None:
            self.paths = self.paths[: int(max_images)]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(resized).float() / 255.0  # [H,W,C] in [0,1]
        x = x.permute(2, 0, 1)  # [3,H,W]
        return x


def compute_mse_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold_percentile: float,
) -> float:
    model.eval()
    mses: List[float] = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            recon = model(x)
            mse = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            mses.extend(mse.tolist())

    if not mses:
        raise RuntimeError("No validation frames to compute threshold.")

    mses_np = np.array(mses, dtype=np.float64)
    thr = float(np.percentile(mses_np, threshold_percentile))
    return thr


def train_datasets(
    train_ds: Dataset,
    val_ds: Dataset,
    out_weights: Path,
    out_threshold: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold_percentile: float,
) -> None:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    out_threshold.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.shape[0]
            train_count += x.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        # Compute validation loss.
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                recon = model(x)
                loss = loss_fn(recon, x)
                val_loss_sum += float(loss.item()) * x.shape[0]
                val_count += x.shape[0]
        val_loss = val_loss_sum / max(1, val_count)

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict()}, str(out_weights))
            print(f"  -> Saved best AE weights to: {out_weights}")

    # Load best weights for threshold computation.
    ckpt = torch.load(str(out_weights), map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    threshold = compute_mse_threshold(
        model=model,
        loader=val_loader,
        device=device,
        threshold_percentile=threshold_percentile,
    )

    out_threshold.write_text(
        json.dumps({"threshold_mse": threshold, "percentile": threshold_percentile}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved anomaly threshold to: {out_threshold} (MSE={threshold:.8f})")


def train(
    train_root: Path,
    val_root: Path,
    out_weights: Path,
    out_threshold: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold_percentile: float,
    max_train_images: Optional[int],
    max_val_images: Optional[int],
) -> None:
    train_ds = FramesDataset(train_root, max_images=max_train_images)
    val_ds = FramesDataset(val_root, max_images=max_val_images)
    if len(train_ds) == 0:
        raise RuntimeError(f"No training frames found in: {train_root}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation frames found in: {val_root}")

    train_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        out_weights=out_weights,
        out_threshold=out_threshold,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold_percentile=threshold_percentile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConvAE for reconstruction-error anomaly detection.")
    parser.add_argument("--train-frames-root", required=False, help="Folder containing normal crowd frames/images.")
    parser.add_argument("--val-frames-root", required=False, help="Folder for threshold calibration (mostly normal).")
    parser.add_argument(
        "--frames-root",
        required=False,
        help="Single frames directory to use for both train and val (auto-split). Overrides train/val roots.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="When using --frames-root, fraction of images used for validation.",
    )
    parser.add_argument("--out-weights", required=True, help="Output checkpoint path (e.g. weights/ae_best.pth).")
    parser.add_argument("--out-threshold", required=True, help="Output JSON path for threshold.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    args = parser.parse_args()

    if args.frames_root:
        frames_root = Path(args.frames_root)
        ds = FramesDataset(frames_root, max_images=None)
        if len(ds) == 0:
            raise RuntimeError(f"No frames found in: {frames_root}")

        val_ratio = float(args.val_ratio)
        n_val = int(len(ds) * val_ratio)
        n_val = max(1, min(len(ds) - 1, n_val)) if len(ds) > 2 else 0
        n_train = len(ds) - n_val
        if n_train <= 0 or n_val <= 0:
            raise RuntimeError("Invalid split sizes. Check --val-ratio.")

        g = torch.Generator().manual_seed(1337)
        train_subset, val_subset = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)

        # Optionally truncate subsets for faster experiments.
        if args.max_train_images is not None:
            n = int(args.max_train_images)
            idx = train_subset.indices[:n]
            train_subset = torch.utils.data.Subset(train_subset.dataset, idx)
        if args.max_val_images is not None:
            n = int(args.max_val_images)
            idx = val_subset.indices[:n]
            val_subset = torch.utils.data.Subset(val_subset.dataset, idx)

        train_datasets(
            train_ds=train_subset,
            val_ds=val_subset,
            out_weights=Path(args.out_weights),
            out_threshold=Path(args.out_threshold),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold_percentile=args.threshold_percentile,
        )
    else:
        if not args.train_frames_root or not args.val_frames_root:
            raise ValueError("Either provide (--train-frames-root & --val-frames-root) or --frames-root.")

        train(
            train_root=Path(args.train_frames_root),
            val_root=Path(args.val_frames_root),
            out_weights=Path(args.out_weights),
            out_threshold=Path(args.out_threshold),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold_percentile=args.threshold_percentile,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
        )


if __name__ == "__main__":
    main()

