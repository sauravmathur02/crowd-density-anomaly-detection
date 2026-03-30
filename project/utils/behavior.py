from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from detector.yolo_detector import Detection


class BehaviorAnalyzer:
    def __init__(self, high_speed_px: float = 35.0) -> None:
        self.high_speed_px = float(high_speed_px)
        self.previous_centers: Dict[int, Tuple[float, float]] = {}
        self.track_ages: Dict[int, int] = {}

    def apply_config(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return
        self.high_speed_px = float(config.get("high_speed_px", self.high_speed_px))

    def reset(self) -> None:
        self.previous_centers.clear()
        self.track_ages.clear()

    def analyze(self, detections: List[Detection]) -> dict:
        current_centers: Dict[int, Tuple[float, float]] = {}
        current_ages: Dict[int, int] = {}
        speeds: Dict[int, float] = {}
        confidence_scale: Dict[int, float] = {}
        suspicious_ids: Set[int] = set()
        weapon_track_ids = {det.track_id for det in detections if det.label == "weapon" and det.track_id is not None}

        for det in detections:
            if det.track_id is None:
                continue

            x1, y1, x2, y2 = det.xyxy
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            current_centers[det.track_id] = center

            age = self.track_ages.get(det.track_id, 0) + 1
            current_ages[det.track_id] = age

            previous = self.previous_centers.get(det.track_id)
            if previous is None:
                speeds[det.track_id] = 0.0
            else:
                speeds[det.track_id] = float(np.linalg.norm(np.array(center) - np.array(previous)))

            confidence_scale[det.track_id] = self._confidence_scale(age)

            if det.track_id in weapon_track_ids and speeds[det.track_id] >= self.high_speed_px:
                suspicious_ids.add(det.track_id)

        self.previous_centers = current_centers
        self.track_ages = current_ages
        return {
            "speeds": speeds,
            "suspicious": bool(suspicious_ids),
            "suspicious_ids": suspicious_ids,
            "confidence_scale": confidence_scale,
            "track_ages": current_ages,
        }

    def _confidence_scale(self, age: int) -> float:
        if age <= 1:
            return 0.90
        if age == 2:
            return 0.96
        if age >= 5:
            return 1.05
        return 1.0
