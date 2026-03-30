from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.yolo_detector import Detection


def get_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    return cap.read()


def get_video_metadata(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Some codecs return 0 fps; fall back to a sane default.
    if fps <= 0:
        fps = 30.0
    return width, height, fps


def make_writer(
    output_path: Optional[str],
    fps: float,
    width: int,
    height: int,
) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Could not open video writer: {output_path}")
    return writer


def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw + 6, y + baseline + 2), bg_color, -1)
    cv2.putText(frame, text, (x + 3, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _point_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def draw_detections(
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    class_id_to_name: Dict[int, str],
    risk_info: dict,
    track_history: dict,
    show_heatmap: bool = True,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw ByteTrack object tracking trajectories, crowd density heatmaps, and unified risk score overlays.
    """
    frame = frame_bgr.copy()
    centers = []

    # Color palette by class id.
    palette = {
        0: (255, 255, 0),    # person - cyan-ish
        1: (0, 0, 255),      # gun - red
        2: (0, 165, 255),    # knife - orange
    }

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cls = det.cls
        color = palette.get(cls, (0, 255, 0))
        
        # Draw bounding boxes and Tracking IDs
        label = f"{class_id_to_name.get(cls, str(cls))} {det.conf:.2f}"
        if det.track_id is not None:
            label += f" | ID:{det.track_id}"
            
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        _put_text_with_bg(
            frame,
            label,
            org=(int(x1), max(0, int(y1))),
            font_scale=font_scale,
            color=color,
            bg_color=(0, 0, 0),
            thickness=2,
        )

        # Center calculation for heatmaps and tracking
        cx, cy = int((x1 + x2) / 2), int(y2)
        if cls == 0:
            centers.append((cx, cy))
            
        # Draw Trajectories if tracked
        if det.track_id is not None and cls == 0:
            track = track_history[det.track_id]
            if track and _point_distance(track[-1], (cx, cy)) > 140:
                track.clear()
            track.append((cx, cy))
            if len(track) > 12:
                track.pop(0)

            if len(track) >= 2:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    # Apply Crowd Density Heatmap Overlay
    if show_heatmap and len(centers) > 0:
        heatmap_layer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for cx, cy in centers:
            # Accumulate Gaussian spots
            cv2.circle(heatmap_layer, (cx, cy), radius=60, color=1.0, thickness=-1)
        
        heatmap_layer = cv2.GaussianBlur(heatmap_layer, (99, 99), 0)
        heatmap_layer = np.clip(heatmap_layer * 255, 0, 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_layer, cv2.COLORMAP_JET)
        
        # Blend Heatmap where density exists
        mask = heatmap_layer > 10
        frame[mask] = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)[mask]

    # Draw Production-Level Risk Score Dashboard Overlay
    score = risk_info.get("score", 0.0)
    severity = risk_info.get("severity", "SAFE")
    
    overlay_color = (0, 204, 0) # Green
    if severity in {"HIGH", "WARNING", "ELEVATED"}:
        overlay_color = (0, 165, 255) # Orange
    elif severity == "CRITICAL":
        overlay_color = (0, 0, 255) # Red

    y = 30
    _put_text_with_bg(
        frame,
        f"INTELLIGENT RISK ENGINE: {score}% [{severity}]",
        (10, y),
        font_scale=0.8,
        color=overlay_color,
        bg_color=(0, 0, 0),
        thickness=2
    )
    y += 30

    if risk_info.get("weapon_detected"):
        _put_text_with_bg(frame, "⚠️ CRITICAL: LETHAL WEAPON DETECTED ⚠️", (10, y), font_scale=0.7, color=(0, 0, 255))
        y += 25
    if risk_info.get("anomaly_mse", 0) > risk_info.get("anomaly_threshold", 1.0):
        _put_text_with_bg(frame, f"⚠️ ABNORMAL BEHAVIOR (TempMSE: {risk_info['anomaly_mse']:.4f})", (10, y), font_scale=0.7, color=(0, 165, 255))
        y += 25
    if risk_info.get("optical_flow_mag", 0) > 2.0:
        _put_text_with_bg(frame, f"⚠️ HIGH CROWD VELOCITY: PANIC DETECTED", (10, y), font_scale=0.7, color=(0, 165, 255))

    active_track_ids = {det.track_id for det in detections if det.track_id is not None}
    for track_id in list(track_history.keys()):
        if track_id not in active_track_ids:
            track_history.pop(track_id, None)

    return frame


