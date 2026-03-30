import cv2
import numpy as np
from typing import Tuple

class MotionAnalyzer:
    """
    Computes dense optical flow (Farneback) to analyze crowd movement.
    Useful for triggering alerts on sudden, abnormal speeds indicative of panic or running.
    """
    def __init__(self, resize_dim: Tuple[int, int] = (320, 240)):
        self.resize_dim = resize_dim
        self.prev_gray = None

    def analyze_motion(self, frame_bgr: np.ndarray, frame_stride: int = 1) -> float:
        """
        Calculates the average motion magnitude of the crowd.
        Returns a flow magnitude scalar (higher = faster movement / panic).
        """
        # Downscale for real-time dense optical flow performance
        small_frame = cv2.resize(frame_bgr, self.resize_dim)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Calculate dense optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.prev_gray,
            next=gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        self.prev_gray = gray

        # Compute magnitude and angle of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # We only care about significantly moving pixels to ignore noise
        moving_pixels = magnitude[magnitude > 2.0]
        
        if len(moving_pixels) == 0:
            return 0.0
            
        # Average velocity of moving pixels
        avg_velocity = float(np.mean(moving_pixels))
        avg_velocity /= max(int(frame_stride), 1)
        return avg_velocity
