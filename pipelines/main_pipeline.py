from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anomaly.anomaly_detector import AnomalyDetector
from detection.yolo_detector import YOLODetector
from utils.video_utils import draw_detections, get_capture, get_video_metadata, make_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crowd density + weapon + anomaly detection pipeline.")

    parser.add_argument("--input-video", required=True, help="Path to input video file.")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional path to write annotated output video (mp4 recommended).",
    )
    parser.add_argument("--display", action="store_true", help="Display the processed video in a window.")

    # YOLOv8 inference settings
    parser.add_argument("--yolo-weights", required=True, help="Path to YOLOv8 weights (e.g., best.pt).")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo-device", default=None, help="YOLO device, e.g. 'cpu' or '0' for GPU index.")

    # Crowd threshold
    parser.add_argument("--crowd-threshold", type=int, default=15, help="High crowd threshold (# people).")

    # Autoencoder anomaly settings
    parser.add_argument(
        "--ae-weights",
        required=True,
        help="Path to ConvAE pretrained weights (checkpoint).",
    )
    parser.add_argument("--ae-device", default="cpu", help="Autoencoder device: 'cpu' or 'cuda'.")
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.01,
        help="Anomaly threshold on reconstruction MSE (needs calibration).",
    )
    parser.add_argument(
        "--ae-threshold-json",
        default=None,
        help="Optional JSON produced by train_conv_ae.py. If set, overrides --anomaly-threshold.",
    )
    parser.add_argument("--ae-every", type=int, default=1, help="Run anomaly detector every N frames.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ae_threshold_json:
        import json

        p = Path(args.ae_threshold_json)
        if not p.exists():
            raise FileNotFoundError(f"--ae-threshold-json not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if "threshold_mse" in payload:
            args.anomaly_threshold = float(payload["threshold_mse"])

    input_video = args.input_video
    if args.output_video:
        Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)

    # Create modules.
    yolo = YOLODetector(
        weights_path=args.yolo_weights,
        device=args.yolo_device,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
    )
    ae = AnomalyDetector(
        weights_path=args.ae_weights,
        device=args.ae_device,
        threshold=args.anomaly_threshold,
        input_size=(256, 256),
    )

    # Video IO.
    cap = get_capture(input_video)
    width, height, fps = get_video_metadata(cap)
    writer = make_writer(args.output_video, fps=fps, width=width, height=height)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO detection.
        detections = yolo.detect(frame)
        people_count = yolo.count_class(detections, 0)
        crowd_high = people_count >= args.crowd_threshold

        weapon_present = yolo.any_class(detections, class_ids=[1, 2])

        # 2) Anomaly detection (reconstruction error).
        anomaly_mse: float = 0.0
        anomaly_flag: bool = False
        if args.ae_every <= 1 or (frame_idx % args.ae_every == 0):
            anomaly_mse, anomaly_flag = ae.score_frame(frame)

        # 3) Draw overlays + output.
        annotated = draw_detections(
            frame_bgr=frame,
            detections=detections,
            class_id_to_name=yolo.class_names,
            alert_weapon=weapon_present,
            crowd_count=people_count,
            crowd_high=crowd_high,
            anomaly_mse=anomaly_mse,
            anomaly_flag=anomaly_flag,
        )

        if writer is not None:
            writer.write(annotated)

        if weapon_present:
            # Print alert message (in addition to overlay).
            print(f"[Frame {frame_idx}] WEAPON ALERT: gun/knife detected")
        if anomaly_flag:
            print(f"[Frame {frame_idx}] ANOMALY DETECTED (MSE={anomaly_mse:.6f})")
        if crowd_high:
            print(f"[Frame {frame_idx}] High Crowd Density (people={people_count})")

        if args.display:
            cv2.imshow("Crowd + Weapons + Anomaly", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

