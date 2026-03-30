from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import load_config, resolve_path
from backend.event_logger import EventLogger
from backend.recording import VideoRecorder
from classifier.weapon_classifier import WeaponClassifier
from detector.yolo_detector import Detection, YOLODetector
from utils.behavior import BehaviorAnalyzer
from utils.risk_engine import compute_risk
from utils.smoothing import DetectionSmoother


@dataclass
class ProcessedFrame:
    source_id: str
    timestamp: str
    detections: List[Detection]
    crowd_count: int
    risk: dict
    suspicious_ids: List[int]
    weapon_labels: List[str]
    annotated_frame: np.ndarray
    alert_active: bool
    detection_count: int
    active_tracks: int
    dominant_weapon: str
    snapshot_path: Optional[str] = None
    recording_path: Optional[str] = None

    def to_response(self) -> dict:
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "crowd_count": self.crowd_count,
            "risk": self.risk,
            "alert_active": self.alert_active,
            "weapon_labels": self.weapon_labels,
            "suspicious_ids": self.suspicious_ids,
            "detection_count": self.detection_count,
            "active_tracks": self.active_tracks,
            "dominant_weapon": self.dominant_weapon,
            "snapshot_path": self.snapshot_path,
            "recording_path": self.recording_path,
            "detections": [detection.to_dict() for detection in self.detections],
        }


@dataclass(frozen=True)
class RenderOptions:
    show_boxes: bool = True
    show_labels: bool = True
    show_tracking: bool = True
    highlight_objects: bool = True


class FeedProcessor:
    def __init__(
        self,
        source_id: str,
        detector: YOLODetector,
        classifier: WeaponClassifier,
        smoother: DetectionSmoother,
        behavior: BehaviorAnalyzer,
        event_logger: Optional[EventLogger] = None,
        recorder: Optional[VideoRecorder] = None,
    ) -> None:
        self.source_id = source_id
        self.detector = detector
        self.classifier = classifier
        self.smoother = smoother
        self.behavior = behavior
        self.event_logger = event_logger
        self.recorder = recorder
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def reset(self) -> None:
        self.detector.reset_tracking()
        self.smoother.reset()
        self.behavior.reset()
        self.track_history.clear()

    def close(self) -> None:
        self.reset()
        if self.recorder is not None:
            self.recorder.release()

    def process_frame(
        self,
        frame_bgr,
        timestamp: Optional[str] = None,
        render_options: Optional[RenderOptions] = None,
    ) -> ProcessedFrame:
        timestamp = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")
        render_options = render_options or RenderOptions()
        detections = self._classify_detections(frame_bgr)
        detections = self.smoother.update(detections)
        behavior_state = self.behavior.analyze(detections)

        for detection in detections:
            if detection.track_id is None:
                continue
            scale = behavior_state["confidence_scale"].get(detection.track_id, 1.0)
            detection.conf = min(1.0, detection.conf * scale)

        crowd_count = self.detector.unique_track_count(detections, "person")
        weapon_labels = [
            detection.weapon_type
            for detection in detections
            if detection.label == "weapon" and detection.weapon_type
        ]
        active_tracks = len({detection.track_id for detection in detections if detection.track_id is not None})
        risk = compute_risk(crowd_count, weapon_labels, behavior_state["suspicious"])

        annotated_frame = draw_annotations(
            frame_bgr.copy(),
            detections,
            self.track_history,
            crowd_count,
            risk,
            behavior_state["suspicious_ids"],
            self.source_id,
            render_options,
        )

        snapshot_path = None
        if self.event_logger is not None:
            snapshot_path = self.event_logger.save_snapshot(
                self.source_id,
                annotated_frame,
                detections,
                timestamp,
            )
            self.event_logger.log_detections(
                self.source_id,
                detections,
                timestamp,
                risk,
                snapshot_path=snapshot_path,
            )

        if self.recorder is not None:
            self.recorder.write(annotated_frame)

        return ProcessedFrame(
            source_id=self.source_id,
            timestamp=timestamp,
            detections=detections,
            crowd_count=crowd_count,
            risk=risk,
            suspicious_ids=sorted(behavior_state["suspicious_ids"]),
            weapon_labels=[label for label in weapon_labels if label],
            annotated_frame=annotated_frame,
            alert_active=bool(weapon_labels),
            detection_count=len(detections),
            active_tracks=active_tracks,
            dominant_weapon=weapon_labels[0] if weapon_labels else "None",
            snapshot_path=snapshot_path,
            recording_path=str(self.recorder.output_path) if self.recorder and self.recorder.output_path else None,
        )

    def _classify_detections(self, frame_bgr) -> List[Detection]:
        detections: List[Detection] = []
        for detection in self.detector.detect(frame_bgr):
            if detection.label != "weapon":
                detections.append(detection)
                continue

            weapon_type, classifier_conf = self.classifier.classify(frame_bgr, detection.xyxy)
            if classifier_conf < self.classifier.confidence_floor:
                continue

            strong_classifier = classifier_conf >= self.classifier.high_conf_override
            min_yolo_conf = (
                self.detector.low_weapon_yolo_conf
                if classifier_conf >= self.classifier.small_object_recovery_conf
                else self.detector.base_weapon_yolo_conf
            )

            if not strong_classifier and detection.detector_conf < min_yolo_conf:
                continue
            if not strong_classifier and detection.area < self.detector.min_weapon_area:
                if classifier_conf < self.classifier.small_object_recovery_conf:
                    continue

            detection.weapon_type = weapon_type
            detection.classifier_conf = classifier_conf
            detection.conf = min(1.0, (0.6 * detection.detector_conf) + (0.4 * classifier_conf))
            detections.append(detection)

        return detections


class RuntimeFactory:
    def __init__(self, root: str | Path, config_path: Optional[str | Path] = None) -> None:
        self.root = Path(root)
        self.config_path = Path(config_path) if config_path else self.root / "config.yaml"
        self.config = load_config(self.config_path)
        self._processors: Dict[str, FeedProcessor] = {}

        self.detector_weights = resolve_path(self.root, self.config["models"]["detector_weights"])
        self.classifier_weights = resolve_path(self.root, self.config["models"]["classifier_weights"])

        self.classifier = WeaponClassifier(self.classifier_weights)
        self.classifier.apply_config(self.config.get("classifier"))

        logging_cfg = self.config.get("logging", {})
        self.event_logger = EventLogger(
            output_dir=resolve_path(self.root, logging_cfg.get("output_dir", "outputs/events")),
            csv_name=str(logging_cfg.get("csv_name", "detections.csv")),
            jsonl_name=str(logging_cfg.get("jsonl_name", "detections.jsonl")),
            snapshots_dir=resolve_path(self.root, logging_cfg.get("snapshots_dir", "outputs/snapshots")),
            enabled=bool(logging_cfg.get("enabled", True)),
            snapshot_cooldown_seconds=float(logging_cfg.get("snapshot_cooldown_seconds", 2.0)),
        )

    def create_feed_processor(
        self,
        source_id: str,
        enable_recording: bool = False,
        fps: float = 20.0,
        persist: bool = False,
    ) -> FeedProcessor:
        detector_cfg = self.config.get("detection", {})
        detector = YOLODetector(
            self.detector_weights,
            conf=float(detector_cfg.get("conf", 0.05)),
            iou=float(detector_cfg.get("iou", 0.70)),
            imgsz=int(detector_cfg.get("imgsz", 960)),
        )
        detector.apply_config(detector_cfg)

        smoother = DetectionSmoother()
        smoother.apply_config(self.config.get("tracking"))
        smoother.immediate_switch_conf = self.classifier.immediate_switch_conf

        behavior = BehaviorAnalyzer()
        behavior.apply_config(self.config.get("behavior"))

        recording_cfg = self.config.get("recording", {})
        recorder = None
        if enable_recording:
            recorder = VideoRecorder(
                output_dir=resolve_path(self.root, recording_cfg.get("output_dir", "outputs/recordings")),
                source_id=source_id,
                fps=fps,
                codec=str(recording_cfg.get("codec", "mp4v")),
                enabled=True,
            )

        processor = FeedProcessor(
            source_id=source_id,
            detector=detector,
            classifier=self.classifier,
            smoother=smoother,
            behavior=behavior,
            event_logger=self.event_logger,
            recorder=recorder,
        )

        if persist:
            self._processors[source_id] = processor
        return processor

    def get_or_create_processor(
        self,
        source_id: str,
        enable_recording: bool = False,
        fps: float = 20.0,
    ) -> FeedProcessor:
        if source_id not in self._processors:
            self._processors[source_id] = self.create_feed_processor(
                source_id=source_id,
                enable_recording=enable_recording,
                fps=fps,
                persist=False,
            )
        return self._processors[source_id]

    def release_processor(self, source_id: str) -> None:
        processor = self._processors.pop(source_id, None)
        if processor is not None:
            processor.close()

    def close(self) -> None:
        for source_id in list(self._processors.keys()):
            self.release_processor(source_id)


def draw_annotations(
    frame_bgr,
    detections: List[Detection],
    track_history: Dict[int, List[Tuple[int, int]]],
    crowd_count: int,
    risk: dict,
    suspicious_ids,
    source_id: str,
    render_options: RenderOptions,
):
    color_map = {
        "person": (40, 220, 110),
        "gun": (60, 60, 255),
        "knife": (0, 165, 255),
        "weapon": (60, 60, 255),
    }

    if not render_options.show_tracking:
        track_history.clear()

    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        label = detection.display_label
        color = color_map.get(label, (0, 255, 0))
        if detection.track_id in suspicious_ids:
            color = (0, 0, 255)

        if render_options.show_boxes:
            if render_options.highlight_objects:
                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.08, frame_bgr, 0.92, 0, frame_bgr)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 4)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        if render_options.show_labels:
            tag = f"{label} {detection.conf:.2f}"
            if render_options.show_tracking and detection.track_id is not None:
                tag += f" | ID:{detection.track_id}"
            if detection.smoothed:
                tag += " | HOLD"

            cv2.rectangle(frame_bgr, (x1, max(0, y1 - 24)), (x1 + 260, y1), (6, 10, 18), -1)
            cv2.putText(
                frame_bgr,
                tag,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if render_options.show_tracking and detection.label == "person" and detection.track_id is not None:
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            history = track_history[detection.track_id]
            history.append(center)
            if len(history) > 20:
                history.pop(0)
            for idx in range(1, len(history)):
                cv2.line(frame_bgr, history[idx - 1], history[idx], color, 2)

    active_ids = {detection.track_id for detection in detections if detection.track_id is not None}
    for track_id in list(track_history.keys()):
        if track_id not in active_ids:
            track_history.pop(track_id, None)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (520, 130), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame_bgr, 0.2, 0, frame_bgr)
    cv2.putText(frame_bgr, f"Source: {source_id}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        f"Risk: {risk['level']} ({risk['score']})",
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Crowd Count: {crowd_count}",
        (20, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Suspicious: {'YES' if risk['suspicious'] else 'NO'}",
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
