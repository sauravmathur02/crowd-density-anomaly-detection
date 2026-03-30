from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from detector.yolo_detector import Detection


@dataclass
class _TrackMemory:
    detection: Detection
    misses: int = 0
    label_history: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    box_history: Deque[Tuple[int, int, int, int]] = field(default_factory=lambda: deque(maxlen=5))
    stable_weapon_type: Optional[str] = None
    pending_weapon_type: Optional[str] = None
    pending_switch_count: int = 0


class DetectionSmoother:
    def __init__(
        self,
        max_missing_frames: int = 3,
        history_size: int = 5,
        switch_confirmations: int = 2,
        static_hold_conf_decay: float = 0.85,
        moving_hold_conf_decay: float = 0.90,
        hold_conf_floor: float = 0.20,
        immediate_switch_conf: float = 0.85,
        moving_threshold_px: float = 12.0,
    ) -> None:
        self.max_missing_frames = max_missing_frames
        self.history_size = history_size
        self.switch_confirmations = switch_confirmations
        self.static_hold_conf_decay = static_hold_conf_decay
        self.moving_hold_conf_decay = moving_hold_conf_decay
        self.hold_conf_floor = hold_conf_floor
        self.immediate_switch_conf = immediate_switch_conf
        self.moving_threshold_px = moving_threshold_px
        self.memory: Dict[int, _TrackMemory] = {}

    def apply_config(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return

        self.max_missing_frames = int(config.get("max_missing_frames", self.max_missing_frames))
        self.history_size = int(config.get("history_size", self.history_size))
        self.switch_confirmations = int(config.get("switch_confirmations", self.switch_confirmations))
        self.static_hold_conf_decay = float(config.get("static_hold_conf_decay", self.static_hold_conf_decay))
        self.moving_hold_conf_decay = float(config.get("moving_hold_conf_decay", self.moving_hold_conf_decay))
        self.hold_conf_floor = float(config.get("hold_conf_floor", self.hold_conf_floor))
        self.immediate_switch_conf = float(config.get("immediate_switch_conf", self.immediate_switch_conf))
        self.moving_threshold_px = float(config.get("moving_threshold_px", self.moving_threshold_px))

    def reset(self) -> None:
        self.memory.clear()

    def update(self, detections: List[Detection]) -> List[Detection]:
        output: List[Detection] = []
        seen_track_ids = set()

        for detection in detections:
            if detection.track_id is None:
                output.append(detection)
                continue

            seen_track_ids.add(detection.track_id)
            memory = self.memory.get(detection.track_id)
            if memory is None:
                memory = _TrackMemory(
                    detection=copy.deepcopy(detection),
                    label_history=deque(maxlen=self.history_size),
                    box_history=deque(maxlen=self.history_size),
                )

            memory.misses = 0
            memory.box_history.append(detection.xyxy)
            detection.xyxy = self._average_box(memory.box_history)

            if detection.label == "weapon" and detection.weapon_type:
                memory.label_history.append(detection.weapon_type)
                detection.weapon_type = self._stable_weapon_label(
                    memory,
                    detection.weapon_type,
                    detection.classifier_conf,
                )

            memory.detection = copy.deepcopy(detection)
            self.memory[detection.track_id] = memory
            output.append(detection)

        for track_id in list(self.memory.keys()):
            if track_id in seen_track_ids:
                continue

            memory = self.memory[track_id]
            memory.misses += 1
            if memory.misses > self.max_missing_frames:
                del self.memory[track_id]
                continue

            held = copy.deepcopy(memory.detection)
            held.smoothed = True
            held.xyxy = self._average_box(memory.box_history)
            held.conf = max(0.0, held.conf * self._ghost_decay(memory.box_history))
            if held.conf < self.hold_conf_floor:
                del self.memory[track_id]
                continue
            if held.label == "weapon":
                held.weapon_type = memory.stable_weapon_type or held.weapon_type
            output.append(held)

        return output

    def _stable_weapon_label(self, memory: _TrackMemory, current_label: str, current_conf: float) -> str:
        if memory.stable_weapon_type is None:
            memory.stable_weapon_type = current_label
            memory.pending_weapon_type = None
            memory.pending_switch_count = 0
            return current_label

        if current_conf >= self.immediate_switch_conf:
            memory.stable_weapon_type = current_label
            memory.pending_weapon_type = None
            memory.pending_switch_count = 0
            return current_label

        if current_label == memory.stable_weapon_type:
            memory.pending_weapon_type = None
            memory.pending_switch_count = 0
            return memory.stable_weapon_type

        if current_label == memory.pending_weapon_type:
            memory.pending_switch_count += 1
        else:
            memory.pending_weapon_type = current_label
            memory.pending_switch_count = 1

        if memory.pending_switch_count >= self.switch_confirmations:
            memory.stable_weapon_type = current_label
            memory.pending_weapon_type = None
            memory.pending_switch_count = 0

        majority_label = self._majority_vote(memory.label_history)
        if majority_label is not None and majority_label == memory.stable_weapon_type:
            return memory.stable_weapon_type
        return memory.stable_weapon_type

    def _majority_vote(self, label_history: Deque[str]) -> Optional[str]:
        if not label_history:
            return None

        counts = Counter(label_history)
        best_count = max(counts.values())
        candidates = {label for label, count in counts.items() if count == best_count}
        for label in reversed(label_history):
            if label in candidates:
                return label
        return None

    def _average_box(self, box_history: Deque[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        if not box_history:
            return (0, 0, 0, 0)

        count = len(box_history)
        x1 = int(round(sum(box[0] for box in box_history) / count))
        y1 = int(round(sum(box[1] for box in box_history) / count))
        x2 = int(round(sum(box[2] for box in box_history) / count))
        y2 = int(round(sum(box[3] for box in box_history) / count))
        return x1, y1, x2, y2

    def _ghost_decay(self, box_history: Deque[Tuple[int, int, int, int]]) -> float:
        if len(box_history) < 2:
            return self.static_hold_conf_decay

        prev = box_history[-2]
        curr = box_history[-1]
        prev_center = ((prev[0] + prev[2]) / 2.0, (prev[1] + prev[3]) / 2.0)
        curr_center = ((curr[0] + curr[2]) / 2.0, (curr[1] + curr[3]) / 2.0)
        delta = ((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2) ** 0.5
        if delta >= self.moving_threshold_px:
            return self.moving_hold_conf_decay
        return self.static_hold_conf_decay
