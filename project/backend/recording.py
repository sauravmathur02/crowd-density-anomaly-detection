from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


class VideoRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        source_id: str,
        fps: float = 20.0,
        codec: str = "mp4v",
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_id = source_id
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self.codec = codec
        self.enabled = enabled
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame_bgr) -> None:
        if not self.enabled:
            return

        if self.writer is None:
            height, width = frame_bgr.shape[:2]
            filename = f"{_safe_name(self.source_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.output_path = self.output_dir / filename
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))

        self.writer.write(frame_bgr)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned or "source"
