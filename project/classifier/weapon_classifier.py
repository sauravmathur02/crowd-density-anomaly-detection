from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class WeaponClassifier:
    def __init__(self, checkpoint_path: str | Path, device: Optional[str] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.available = False
        self.input_size = 224
        self.labels = ["gun", "knife"]
        self.confidence_floor = 0.55
        self.small_object_recovery_conf = 0.75
        self.high_conf_override = 0.80
        self.immediate_switch_conf = 0.85
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.model = self._build_model(num_classes=len(self.labels))
        self.model.to(self.device)
        self.model.eval()

        self._try_load()

    def apply_config(self, config: Optional[Dict[str, Any]]) -> None:
        if not config:
            return

        self.confidence_floor = float(config.get("confidence_floor", self.confidence_floor))
        self.small_object_recovery_conf = float(
            config.get("small_object_recovery_conf", self.small_object_recovery_conf)
        )
        self.high_conf_override = float(config.get("high_conf_override", self.high_conf_override))
        self.immediate_switch_conf = float(config.get("immediate_switch_conf", self.immediate_switch_conf))

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    def _try_load(self) -> None:
        if not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size == 0:
            self.available = False
            return

        try:
            checkpoint = self._load_checkpoint()
            if isinstance(checkpoint, dict):
                labels = checkpoint.get("labels")
                input_size = checkpoint.get("input_size")
                if labels and isinstance(labels, list):
                    self.labels = [str(label) for label in labels]
                    self.model = self._build_model(num_classes=len(self.labels))
                    self.model.to(self.device)
                if input_size:
                    self.input_size = int(input_size)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def _load_checkpoint(self):
        try:
            return torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(str(self.checkpoint_path), map_location=self.device)

    @torch.inference_mode()
    def classify(self, frame_bgr: np.ndarray, xyxy: Tuple[int, int, int, int]) -> Tuple[str, float]:
        x1, y1, x2, y2 = xyxy
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return "gun", 0.0

        if not self.available:
            raise RuntimeError(f"Classifier checkpoint not available: {self.checkpoint_path}")

        x = self._preprocess(roi)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs).item())
        return self.labels[idx], float(probs[idx].item())

    def _preprocess(self, roi_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        side = max(rgb.shape[:2])
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y_off = (side - rgb.shape[0]) // 2
        x_off = (side - rgb.shape[1]) // 2
        canvas[y_off:y_off + rgb.shape[0], x_off:x_off + rgb.shape[1]] = rgb
        resized = cv2.resize(canvas, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor
