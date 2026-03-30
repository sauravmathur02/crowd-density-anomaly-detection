from __future__ import annotations

import tempfile
from pathlib import Path


_BYTETRACK_CONFIG = """tracker_type: bytetrack
track_high_thresh: 0.30
track_low_thresh: 0.05
new_track_thresh: 0.30
track_buffer: 50
match_thresh: 0.80
fuse_score: True
"""


def get_bytetrack_config_path() -> str:
    """
    Create a custom ByteTrack config for Ultralytics at runtime.
    """
    path = Path(tempfile.gettempdir()) / "production_surveillance_bytetrack.yaml"
    if not path.exists() or path.read_text(encoding="utf-8") != _BYTETRACK_CONFIG:
        path.write_text(_BYTETRACK_CONFIG, encoding="utf-8")
    return str(path)
