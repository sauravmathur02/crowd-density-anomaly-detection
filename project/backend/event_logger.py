from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2

from detector.yolo_detector import Detection


class EventLogger:
    def __init__(
        self,
        output_dir: str | Path,
        csv_name: str,
        jsonl_name: str,
        snapshots_dir: str | Path,
        enabled: bool = True,
        snapshot_cooldown_seconds: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.csv_path = self.output_dir / csv_name
        self.jsonl_path = self.output_dir / jsonl_name
        self.snapshot_cooldown_seconds = float(snapshot_cooldown_seconds)
        self._lock = threading.Lock()
        self._last_snapshot_at: dict[str, float] = {}
        self._csv_headers = [
            "timestamp",
            "source_id",
            "class",
            "base_label",
            "confidence",
            "detector_confidence",
            "classifier_confidence",
            "track_id",
            "bbox",
            "risk_level",
            "risk_score",
            "snapshot_path",
        ]

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_headers)
            writer.writeheader()

    def log_detections(
        self,
        source_id: str,
        detections: Iterable[Detection],
        timestamp: str,
        risk: dict,
        snapshot_path: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        rows = []
        for detection in detections:
            rows.append(
                {
                    "timestamp": timestamp,
                    "source_id": source_id,
                    "class": detection.display_label,
                    "base_label": detection.label,
                    "confidence": round(float(detection.conf), 4),
                    "detector_confidence": round(float(detection.detector_conf), 4),
                    "classifier_confidence": round(float(detection.classifier_conf), 4),
                    "track_id": detection.track_id,
                    "bbox": list(detection.xyxy),
                    "risk_level": risk.get("level"),
                    "risk_score": risk.get("score"),
                    "snapshot_path": snapshot_path,
                }
            )

        if not rows:
            return

        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as csv_handle:
                writer = csv.DictWriter(csv_handle, fieldnames=self._csv_headers)
                writer.writerows(rows)

            with self.jsonl_path.open("a", encoding="utf-8") as json_handle:
                for row in rows:
                    json_handle.write(json.dumps(row) + "\n")

    def save_snapshot(
        self,
        source_id: str,
        annotated_frame,
        detections: Iterable[Detection],
        timestamp: str,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        if not any(detection.label == "weapon" for detection in detections):
            return None

        now = time.time()
        last = self._last_snapshot_at.get(source_id, 0.0)
        if now - last < self.snapshot_cooldown_seconds:
            return None

        snapshot_name = f"{_safe_name(source_id)}_{_timestamp_slug(timestamp)}.jpg"
        snapshot_path = self.snapshots_dir / snapshot_name
        with self._lock:
            cv2.imwrite(str(snapshot_path), annotated_frame)
            self._last_snapshot_at[source_id] = now
        return str(snapshot_path)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "source"


def _timestamp_slug(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%Y%m%d_%H%M%S")
    except ValueError:
        return _safe_name(value)
