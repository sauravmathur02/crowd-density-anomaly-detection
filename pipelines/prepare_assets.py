from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_python(module_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(module_path)] + args
    print("\nRunning:\n  " + " ".join([f"\"{c}\"" if " " in c else c for c in cmd]) + "\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO + ConvAE weights for the main pipeline.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]), help="Repo root.")

    parser.add_argument("--coco-max-train-images", type=int, default=None, help="Optional COCO train image limit (for faster smoke tests).")
    parser.add_argument("--coco-max-val-images", type=int, default=None, help="Optional COCO val image limit (for faster smoke tests).")
    parser.add_argument("--yolo-epochs", type=int, default=50)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--crowd-threshold", type=int, default=15)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--ae-device", default="cpu")
    parser.add_argument("--ae-threshold-percentile", type=float, default=95.0)

    parser.add_argument("--skip-train", action="store_true", help="Only build datasets / print commands, don't train.")
    parser.add_argument("--force-rebuild-combined-dataset", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    data_root = root / "data"
    gun_dataset = root / "gun_dataset"
    coco_root = data_root / "COCO" / "coco2017"
    knife_images_dir = data_root / "Knife" / "knife"

    combined_out = root / "assets" / "combined_yolo_dataset"
    weights_dir = root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    yolo_weights_out = weights_dir / "best.pt"
    ae_weights_out = weights_dir / "ae_best.pth"
    ae_threshold_json = weights_dir / "ae_threshold.json"

    combined_yaml = combined_out / "data.yaml"

    # 1) Build combined dataset (if missing).
    if args.force_rebuild_combined_dataset or not combined_yaml.exists():
        if combined_out.exists():
            shutil.rmtree(combined_out)
        combined_out.mkdir(parents=True, exist_ok=True)

        builder = root / "data_preparation" / "build_combined_yolo_dataset.py"
        run_python(
            builder,
            [
                "--out-dir",
                str(combined_out),
                "--coco-root",
                str(coco_root),
                "--gun-root",
                str(gun_dataset),
                "--knife-images-dir",
                str(knife_images_dir),
                *(["--coco-max-train-images", str(args.coco_max_train_images)] if args.coco_max_train_images else []),
                *(["--coco-max-val-images", str(args.coco_max_val_images)] if args.coco_max_val_images else []),
            ],
        )
    else:
        print(f"Combined YOLO dataset already exists: {combined_yaml}")

    # 2) Train YOLO if weights missing.
    if not yolo_weights_out.exists():
        if args.skip_train:
            print("Skipping YOLO training because --skip-train was set.")
        else:
            train_yolo = root / "pipelines" / "train_yolo_detector.py"
            run_python(
                train_yolo,
                [
                    "--data-yaml",
                    str(combined_yaml),
                    "--imgsz",
                    str(args.yolo_imgsz),
                    "--epochs",
                    str(args.yolo_epochs),
                    "--out-weights",
                    str(yolo_weights_out),
                ],
            )
    else:
        print(f"YOLO weights already exist: {yolo_weights_out}")

    # 3) Train ConvAE if weights missing.
    if not ae_weights_out.exists() or not ae_threshold_json.exists():
        if args.skip_train:
            print("Skipping ConvAE training because --skip-train was set.")
        else:
            train_ae = root / "pipelines" / "train_conv_ae.py"
            # Use extracted crowd frames for anomaly training & threshold calibration.
            # This dataset is unsupervised: threshold is set by percentile of reconstruction error.
            train_frames_root = data_root / "shanghaitech" / "testing" / "frames"
            if not train_frames_root.exists():
                raise FileNotFoundError(f"ConvAE frames-root not found: {train_frames_root}")

            run_python(
                train_ae,
                [
                    "--frames-root",
                    str(train_frames_root),
                    "--val-ratio",
                    "0.2",
                    "--out-weights",
                    str(ae_weights_out),
                    "--out-threshold",
                    str(ae_threshold_json),
                    "--device",
                    args.ae_device,
                    "--epochs",
                    str(args.ae_epochs),
                    "--threshold-percentile",
                    str(args.ae_threshold_percentile),
                ],
            )
    else:
        print(f"ConvAE weights already exist: {ae_weights_out}")
        print(f"ConvAE threshold already exist: {ae_threshold_json}")

    # 4) Print final main_pipeline command.
    # main_pipeline.py supports --ae-threshold-json to override the threshold automatically.
    final_cmd = [
        sys.executable,
        str(root / "pipelines" / "main_pipeline.py"),
        "--input-video",
        '"<INPUT_VIDEO_PATH>"',
        "--output-video",
        '"<OUTPUT_VIDEO_PATH>"',
        "--display",
        "--yolo-weights",
        f'"{yolo_weights_out}"',
        "--ae-weights",
        f'"{ae_weights_out}"',
        "--ae-threshold-json",
        f'"{ae_threshold_json}"',
        "--crowd-threshold",
        str(args.crowd_threshold),
    ]
    print("\nFinal command to run the full system:\n")
    print(" ".join(final_cmd).replace('"<INPUT_VIDEO_PATH>"', '<INPUT_VIDEO_PATH>').replace('"<OUTPUT_VIDEO_PATH>"', '<OUTPUT_VIDEO_PATH>'))


if __name__ == "__main__":
    main()

