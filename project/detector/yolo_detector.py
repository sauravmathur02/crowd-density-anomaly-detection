from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from detector.tracker import get_bytetrack_config_path


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    area: int
    track_id: Optional[int]
    source_class_id: int
    weapon_type: Optional[str] = None
    detector_conf: float = 0.0
    classifier_conf: float = 0.0
    smoothed: bool = False

    @property
    def display_label(self) -> str:
        if self.label == "weapon" and self.weapon_type:
            return self.weapon_type
        return self.label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "display_label": self.display_label,
            "confidence": round(float(self.conf), 4),
            "detector_confidence": round(float(self.detector_conf), 4),
            "classifier_confidence": round(float(self.classifier_conf), 4),
            "track_id": self.track_id,
            "bbox": list(self.xyxy),
            "area": int(self.area),
            "source_class_id": int(self.source_class_id),
            "weapon_type": self.weapon_type,
            "smoothed": bool(self.smoothed),
        }


class YOLODetector:
    def __init__(
        self,
        weights_path: str | Path,
        device: Optional[str] = None,
        conf: float = 0.05,
        iou: float = 0.70,
        imgsz: int = 960,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = str(weights_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_half = self.device != "cpu"
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)

        self.person_threshold = 0.20
        self.weapon_threshold = 0.20
        self.min_person_area = 8000
        self.min_weapon_area = 3500
        self.min_recoverable_weapon_area = 1600
        self.min_box_dim = 40
        self.min_person_yolo_conf = 0.20
        self.min_weapon_candidate_conf = 0.15
        self.base_weapon_yolo_conf = 0.20
        self.low_weapon_yolo_conf = 0.15
        self.weapon_aspect_max = 6.0
        self.weapon_aspect_min = 0.10
        self.top_ignore_ratio = 0.05
        self.border_ignore_ratio = 0.02

        self.model = YOLO(self.weights_path)
        self.tracker_config = get_bytetrack_config_path()

    def apply_config(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return

        self.person_threshold = float(config.get("person_threshold", self.person_threshold))
        self.weapon_threshold = float(config.get("weapon_threshold", self.weapon_threshold))
        self.min_person_area = int(config.get("min_person_area", self.min_person_area))
        self.min_weapon_area = int(config.get("min_weapon_area", self.min_weapon_area))
        self.min_recoverable_weapon_area = int(
            config.get("min_recoverable_weapon_area", self.min_recoverable_weapon_area)
        )
        self.min_box_dim = int(config.get("min_box_dim", self.min_box_dim))
        self.min_person_yolo_conf = float(config.get("min_person_yolo_conf", self.min_person_yolo_conf))
        self.min_weapon_candidate_conf = float(
            config.get("min_weapon_candidate_conf", self.min_weapon_candidate_conf)
        )
        self.base_weapon_yolo_conf = float(config.get("base_weapon_yolo_conf", self.base_weapon_yolo_conf))
        self.low_weapon_yolo_conf = float(config.get("low_weapon_yolo_conf", self.low_weapon_yolo_conf))
        self.weapon_aspect_max = float(config.get("weapon_aspect_max", self.weapon_aspect_max))
        self.weapon_aspect_min = float(config.get("weapon_aspect_min", self.weapon_aspect_min))
        self.top_ignore_ratio = float(config.get("top_ignore_ratio", self.top_ignore_ratio))
        self.border_ignore_ratio = float(config.get("border_ignore_ratio", self.border_ignore_ratio))

    def reset_tracking(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                if hasattr(tracker, "reset"):
                    tracker.reset()

    def detect(self, frame_bgr) -> List[Detection]:
        results = self.model.track(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker_config,
            device=self.device,
            classes=[0, 1, 2],
            half=self.use_half,
            verbose=False,
            max_det=400,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        detections: List[Detection] = []

        for box in result.boxes:
            raw_cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area = box_w * box_h
            aspect_ratio = float(box_w) / float(max(box_h, 1))
            conf = float(box.conf[0].item())

            if box_w < self.min_box_dim or box_h < self.min_box_dim:
                continue

            if raw_cls_id == 0:
                if conf < self.min_person_yolo_conf:
                    continue
                if area < self.min_person_area:
                    continue
                if self._is_fragmented_person(box_w, box_h, area):
                    continue
                if not self._passes_person_threshold(conf, area):
                    continue
                conf = self._boost_person_confidence(conf, area)
                label = "person"
                source_class_id = 0
            else:
                if conf < self.min_weapon_candidate_conf:
                    continue
                if area < self.min_recoverable_weapon_area:
                    continue
                if aspect_ratio > self.weapon_aspect_max or aspect_ratio < self.weapon_aspect_min:
                    continue
                conf *= self._weapon_region_penalty(x1, y1, x2, y2, frame_w, frame_h)
                if not self._passes_weapon_threshold(conf, area):
                    continue
                label = "weapon"
                source_class_id = 1

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            detections.append(
                Detection(
                    label=label,
                    conf=conf,
                    xyxy=(x1, y1, x2, y2),
                    area=area,
                    track_id=track_id,
                    source_class_id=source_class_id,
                    detector_conf=conf,
                )
            )

        return detections

    def _passes_person_threshold(self, conf: float, area: int) -> bool:
        threshold = self.person_threshold
        if area < 15000:
            threshold = 0.26
        elif area < 30000:
            threshold = 0.22
        threshold = max(0.15, threshold)
        return conf >= threshold

    def _passes_weapon_threshold(self, conf: float, area: int) -> bool:
        threshold = self.weapon_threshold
        if area < self.min_weapon_area:
            threshold = max(threshold, 0.22)
        threshold = max(0.15, threshold)
        return conf >= threshold

    def _is_fragmented_person(self, box_w: int, box_h: int, area: int) -> bool:
        if box_h < 80:
            return True
        aspect_ratio = float(box_w) / float(max(box_h, 1))
        if aspect_ratio < 0.30:
            return True
        if aspect_ratio > 1.35 and area < 18000:
            return True
        return False

    def _boost_person_confidence(self, conf: float, area: int) -> float:
        if area >= 90000:
            return min(1.0, conf * 1.10)
        if area >= 50000:
            return min(1.0, conf * 1.05)
        return conf

    def _weapon_region_penalty(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
    ) -> float:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        penalty = 1.0
        if center_y < frame_h * self.top_ignore_ratio:
            penalty *= 0.8
        if center_x < frame_w * self.border_ignore_ratio:
            penalty *= 0.8
        if center_x > frame_w * (1.0 - self.border_ignore_ratio):
            penalty *= 0.8
        return penalty

    def unique_track_count(self, detections: Sequence[Detection], label: str) -> int:
        return len({det.track_id for det in detections if det.label == label and det.track_id is not None})
