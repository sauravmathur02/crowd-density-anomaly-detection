from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for person+gun+knife.")
    parser.add_argument("--data-yaml", required=True, help="Ultralytics data.yaml path for the combined dataset.")
    parser.add_argument("--weights-init", default="yolov8m.pt", help="Initial weights (e.g. yolov8n.pt).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. Use -1 for AutoBatch (adaptive).")
    parser.add_argument("--device", default=None, help="Ultralytics device (e.g. '0' or 'cpu').")
    parser.add_argument("--out-weights", required=True, help="Where to copy the best.pt file.")
    args = parser.parse_args()

    from ultralytics import YOLO

    out_weights = Path(args.out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights_init)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=str(Path(args.data_yaml)),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        workers=2,
        patience=50,  # Early stopping patience
        # Integrated Phase 1 Data Augmentations
        mosaic=1.0, 
        mixup=0.1, 
        copy_paste=0.05,
    )

    # Ultralytics writes into runs/detect/train*/weights/best.pt
    run_dir = None
    # `results` can be an object; easiest is to search newest best.pt under runs.
    runs_dir = Path("runs")  # relative to current working dir
    if not runs_dir.exists():
        # Fallback to Ultralytics default where it may create runs under repo root.
        runs_dir = Path.cwd() / "runs"

    candidates = list(runs_dir.glob("detect/train*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("Could not find runs/detect/train*/weights/best.pt after training.")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best_pt = candidates[0]
    shutil.copy2(best_pt, out_weights)

    print(f"YOLO training done. best weights copied to: {out_weights}")


if __name__ == "__main__":
    main()

