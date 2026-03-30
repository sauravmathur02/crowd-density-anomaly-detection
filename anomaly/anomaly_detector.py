from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from models.autoencoder import ConvAE
from models.convlstm import TemporalConvLSTM


class AnomalyDetector:
    """
    Reconstruction-error anomaly detector using the project's ConvAE.

    Input:
      - OpenCV frame (BGR)
      - resized to 256x256
      - normalized to [0, 1]

    Output:
      - reconstruction MSE score
      - boolean anomaly flag using a threshold
    """

    def __init__(
        self,
        weights_path: Optional[str],
        device: str = "cpu",
        threshold: float = 0.01,
        input_size: Tuple[int, int] = (256, 256),
        temporal: bool = False,
        seq_length: int = 8,
    ) -> None:
        self.device = device
        self.threshold = float(threshold)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        
        self.temporal = temporal
        self.seq_length = seq_length
        self.frame_buffer = []
        self.current_threshold = float(threshold)
        self.enable_adaptive_threshold = True
        self._baseline_warmup = 20
        self._score_count = 0
        self._score_mean = 0.0
        self._score_var = 0.0

        if self.temporal:
            self.model = TemporalConvLSTM().to(self.device)
        else:
            self.model = ConvAE().to(self.device)
            
        if weights_path:
            try:
                self._load_weights(weights_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not load weights for Anomaly Model (possibly missing ConvLSTM weights). Starting untrained. Exception: {e}")
        self.model.eval()

    def _load_weights(self, weights_path: str) -> None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"Autoencoder weights not found: {weights_path}")

        checkpoint = torch.load(str(p), map_location=self.device)
        # Support common checkpoint formats.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assume the dict itself is a state_dict.
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Some training code prefixes keys with "module." (DDP). Strip if needed.
        normalized_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                normalized_state_dict[k[len("module.") :]] = v
            else:
                normalized_state_dict[k] = v

        # The app may request temporal mode while only ConvAE weights are present.
        # Falling back here avoids loading incompatible weights into a random ConvLSTM.
        if self.temporal and not any(k.startswith("convlstm.") for k in normalized_state_dict):
            self.temporal = False
            self.frame_buffer = []
            self.model = ConvAE().to(self.device)

        self.model.load_state_dict(normalized_state_dict, strict=False)

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # Convert BGR -> RGB, then resize.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # [H, W, C] uint8 -> [1, 3, H, W] float32 in [0, 1]
        x = torch.from_numpy(resized).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def score_frame(self, frame_bgr: np.ndarray) -> Tuple[float, bool]:
        x = self.preprocess(frame_bgr)
        
        if self.temporal:
            # Maintain a rolling buffer of frames
            if len(self.frame_buffer) < self.seq_length:
                self.frame_buffer.append(x)
                if len(self.frame_buffer) < self.seq_length:
                    return 0.0, False # Not enough frames to score
                    
            self.frame_buffer.pop(0)
            self.frame_buffer.append(x)
            
            # Stack into [1, seq_len, C, H, W]
            seq_tensor = torch.stack(self.frame_buffer, dim=1)
            recon = self.model(seq_tensor)
            
            # Compare latest frame with latest reconstruction
            latest_recon = recon[:, -1, ...]
            latest_orig = seq_tensor[:, -1, ...]
            mse = torch.mean((latest_recon - latest_orig) ** 2).item()
            
        else:
            recon = self.model(x)
            mse = torch.mean((recon - x) ** 2).item()

        self._update_baseline(mse)
        effective_threshold = self._effective_threshold()
        self.current_threshold = effective_threshold
        is_anomaly = self._score_count > self._baseline_warmup and mse > effective_threshold
        return mse, is_anomaly

    def _update_baseline(self, mse: float) -> None:
        self._score_count += 1

        if self._score_count == 1:
            self._score_mean = mse
            self._score_var = 0.0
            return

        alpha = 0.05
        delta = mse - self._score_mean
        self._score_mean = ((1.0 - alpha) * self._score_mean) + (alpha * mse)
        self._score_var = ((1.0 - alpha) * self._score_var) + (alpha * (delta ** 2))

    def _effective_threshold(self) -> float:
        baseline_std = float(np.sqrt(max(self._score_var, 0.0)))
        adaptive_threshold = self._score_mean + (3.0 * baseline_std)

        if not self.enable_adaptive_threshold:
            return self.threshold

        # Protect against placeholder thresholds such as 0.01 that are not calibrated.
        if self.threshold <= 0.02:
            return max(adaptive_threshold, self._score_mean * 1.35, 0.08)

        return max(self.threshold, adaptive_threshold)

