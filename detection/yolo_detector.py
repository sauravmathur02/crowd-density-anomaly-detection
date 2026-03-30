from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection result."""

    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    track_id: Optional[int] = None


@dataclass
class _TrackState:
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]
    misses: int = 0
    hits: int = 1


class YOLODetector:
    """
    Ultralytics YOLOv8 inference wrapper.

    Expected class mapping (fixed by your requirements):
      0 = person
      1 = gun
      2 = knife
    """

    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        conf: float = 0.08,
        iou: float = 0.72,
        imgsz: int = 960,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.device = self._resolve_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.use_half = self.device != "cpu"

        default_names = {0: "person", 1: "gun", 2: "knife"}
        self.class_names: Dict[int, str] = class_names or default_names

        self.person_conf = 0.18
        self.weapon_conf = 0.24
        self.match_iou = 0.35
        self.nms_iou = 0.65
        self.smoothing_alpha = 0.65
        self.persistence_frames = {0: 3, 1: 2, 2: 2}
        self._next_track_id = 1
        self._active_tracks: Dict[int, _TrackState] = {}

        self.model = YOLO(self.weights_path)

    def _resolve_device(self, device: Optional[str]) -> str:
        if device:
            return str(device)

        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def reset_tracking(self) -> None:
        self._active_tracks.clear()
        self._next_track_id = 1

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                if hasattr(tracker, "reset"):
                    tracker.reset()

    def detect(self, frame_bgr: np.ndarray, persist: bool = True) -> List[Detection]:
        """
        Run YOLO inference on a single frame and stabilize the results over time.
        """
        infer_kwargs = {
            "source": frame_bgr,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": [0, 1, 2],
            "agnostic_nms": True,
            "max_det": 1000,
            "half": self.use_half,
            "verbose": False,
        }

        # Use plain detector outputs and apply our own temporal tracking.
        # This avoids tracker drift/ID jumps when the app skips frames aggressively.
        results = self.model.predict(**infer_kwargs)

        parsed = self._parse_results(results, frame_bgr.shape)
        filtered = self._class_agnostic_nms(parsed)
        if not persist:
            return filtered
        return self._stabilize(filtered)

    def _parse_results(
        self,
        results: Sequence[object],
        frame_shape: Tuple[int, ...],
    ) -> List[Detection]:
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        dets: List[Detection] = []
        for box in r0.boxes:
            xyxy = tuple(float(v) for v in box.xyxy[0].detach().cpu().numpy().tolist())
            cls = int(box.cls[0].detach().cpu().numpy().item())
            conf = float(box.conf[0].detach().cpu().numpy().item())

            if cls not in (0, 1, 2):
                continue

            cls = self._correct_person_misclassification(cls, conf, xyxy, frame_shape)
            if cls is None:
                continue
            cls = self._correct_weapon_class(cls, xyxy, frame_shape)
            if not self._passes_threshold(cls, conf, xyxy, frame_shape):
                continue

            track_id = None
            if getattr(box, "id", None) is not None:
                track_id = int(box.id[0].detach().cpu().numpy().item())

            dets.append(Detection(cls=cls, conf=conf, xyxy=xyxy, track_id=track_id))
        return dets

    def _correct_person_misclassification(
        self,
        cls: int,
        conf: float,
        xyxy: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
    ) -> Optional[int]:
        if cls != 0:
            return cls

        x1, y1, x2, y2 = xyxy
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        aspect_ratio = box_w / box_h
        frame_h, frame_w = frame_shape[:2]
        area_ratio = (box_w * box_h) / max(float(frame_h * frame_w), 1.0)

        # Horizontal compact boxes in this project are usually weapons or props,
        # not full human bodies.
        if aspect_ratio >= 1.6 and box_h <= frame_h * 0.42 and area_ratio <= 0.02:
            return 1

        # Tiny/slender vertical boxes can be knives rather than persons.
        if aspect_ratio <= 0.38 and box_h <= frame_h * 0.30 and area_ratio <= 0.004:
            return 2

        # Suppress implausible person boxes instead of letting them pollute
        # tracking and heatmaps.
        if box_h < 40 and conf < 0.45:
            return None
        if aspect_ratio > 1.15 and conf < 0.60:
            return None
        if area_ratio < 0.0012 and conf < 0.42:
            return None

        return 0

    def _passes_threshold(
        self,
        cls: int,
        conf: float,
        xyxy: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
    ) -> bool:
        if cls == 0:
            return conf >= self.person_conf

        frame_area = max(float(frame_shape[0] * frame_shape[1]), 1.0)
        area_ratio = self._area(xyxy) / frame_area
        adaptive_weapon_conf = self.weapon_conf

        # Small weapons are often slightly under-confident; give them a little room
        # without dropping into the base model noise floor.
        if area_ratio < 0.0015:
            adaptive_weapon_conf = 0.20

        return conf >= adaptive_weapon_conf

    def _correct_weapon_class(
        self,
        cls: int,
        xyxy: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
    ) -> int:
        if cls not in (1, 2):
            return cls

        x1, y1, x2, y2 = xyxy
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        aspect_ratio = box_w / box_h
        frame_area = max(float(frame_shape[0] * frame_shape[1]), 1.0)
        area_ratio = (box_w * box_h) / frame_area

        # Guns tend to be wider and slightly larger in image area.
        if cls == 2 and aspect_ratio >= 1.45 and area_ratio >= 0.00015:
            return 1

        # Knives tend to be slimmer/taller and often occupy less area.
        if cls == 1 and aspect_ratio <= 0.42 and area_ratio <= 0.0025:
            return 2

        return cls

    def _class_agnostic_nms(self, detections: Sequence[Detection]) -> List[Detection]:
        if not detections:
            return []

        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            if all(self._iou(det.xyxy, prev.xyxy) < self.nms_iou for prev in kept):
                kept.append(det)
        return kept

    def _stabilize(self, detections: Sequence[Detection]) -> List[Detection]:
        updated_tracks: Dict[int, _TrackState] = {}
        outputs: List[Detection] = []
        remaining_track_ids = set(self._active_tracks.keys())

        for det in detections:
            matched_id = self._resolve_track_id(det, remaining_track_ids)
            previous = self._active_tracks.get(matched_id) if matched_id is not None else None

            if matched_id is None:
                matched_id = self._next_track_id
                self._next_track_id += 1

            if previous is not None:
                remaining_track_ids.discard(matched_id)
                stable_cls = self._stabilize_class(previous, det)
                stable_conf = max(det.conf, previous.conf * 0.80)
                stable_box = self._smooth_box(previous.xyxy, det.xyxy)
                state = _TrackState(
                    cls=stable_cls,
                    conf=stable_conf,
                    xyxy=stable_box,
                    misses=0,
                    hits=previous.hits + 1,
                )
            else:
                state = _TrackState(
                    cls=det.cls,
                    conf=det.conf,
                    xyxy=det.xyxy,
                    misses=0,
                    hits=1,
                )

            updated_tracks[matched_id] = state
            outputs.append(
                Detection(
                    cls=state.cls,
                    conf=state.conf,
                    xyxy=state.xyxy,
                    track_id=matched_id,
                )
            )

        for track_id in list(remaining_track_ids):
            previous = self._active_tracks[track_id]
            misses = previous.misses + 1
            if misses > self.persistence_frames.get(previous.cls, 2):
                continue

            decayed_conf = previous.conf * (0.90 if previous.cls == 0 else 0.87)
            floor = self.person_conf if previous.cls == 0 else self.weapon_conf
            decayed_conf = max(decayed_conf, floor)

            state = _TrackState(
                cls=previous.cls,
                conf=decayed_conf,
                xyxy=previous.xyxy,
                misses=misses,
                hits=previous.hits,
            )
            updated_tracks[track_id] = state
            outputs.append(
                Detection(
                    cls=state.cls,
                    conf=state.conf,
                    xyxy=state.xyxy,
                    track_id=track_id,
                )
            )

        self._active_tracks = updated_tracks
        return sorted(outputs, key=lambda det: (det.cls, -(det.conf), det.track_id or -1))

    def _resolve_track_id(
        self,
        det: Detection,
        remaining_track_ids: Iterable[int],
    ) -> Optional[int]:
        if det.track_id is not None and det.track_id in self._active_tracks:
            return det.track_id

        best_track_id = None
        best_iou = 0.0
        for track_id in remaining_track_ids:
            state = self._active_tracks[track_id]
            if not self._compatible_classes(state.cls, det.cls):
                continue

            overlap = self._iou(state.xyxy, det.xyxy)
            if overlap >= self.match_iou and overlap > best_iou:
                best_iou = overlap
                best_track_id = track_id

        if best_track_id is not None:
            return best_track_id

        return det.track_id

    def _compatible_classes(self, previous_cls: int, current_cls: int) -> bool:
        if previous_cls == current_cls:
            return True
        return previous_cls in (1, 2) and current_cls in (1, 2)

    def _stabilize_class(self, previous: _TrackState, current: Detection) -> int:
        if previous.cls == current.cls:
            return current.cls

        # Weapons are allowed to change class only with clearly stronger evidence.
        # This reduces frame-to-frame gun/knife flipping.
        if previous.cls in (1, 2) and current.cls in (1, 2):
            if current.conf >= previous.conf + 0.18:
                return current.cls
            return previous.cls

        if current.conf >= max(previous.conf + 0.10, 0.60):
            return current.cls
        return previous.cls

    def _smooth_box(
        self,
        previous_xyxy: Tuple[float, float, float, float],
        current_xyxy: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        alpha = self.smoothing_alpha
        return tuple(
            (alpha * current) + ((1.0 - alpha) * previous)
            for previous, current in zip(previous_xyxy, current_xyxy)
        )

    def _area(self, xyxy: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _iou(
        self,
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        union = self._area(box_a) + self._area(box_b) - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def count_class(self, detections: Sequence[Detection], class_id: int) -> int:
        return sum(1 for d in detections if d.cls == class_id)

    def any_class(self, detections: Sequence[Detection], class_ids: Iterable[int]) -> bool:
        class_ids_set = set(class_ids)
        return any(d.cls in class_ids_set for d in detections)

    def class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, str(class_id))
