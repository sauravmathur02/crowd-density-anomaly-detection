from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "detector_weights": "models/best.pt",
        "classifier_weights": "models/classifier.pth",
    },
    "detection": {
        "conf": 0.05,
        "iou": 0.70,
        "imgsz": 960,
        "person_threshold": 0.20,
        "weapon_threshold": 0.20,
        "min_person_area": 8000,
        "min_weapon_area": 3500,
        "min_recoverable_weapon_area": 1600,
        "min_box_dim": 40,
        "min_person_yolo_conf": 0.20,
        "min_weapon_candidate_conf": 0.15,
        "base_weapon_yolo_conf": 0.20,
        "low_weapon_yolo_conf": 0.15,
        "weapon_aspect_max": 6.0,
        "weapon_aspect_min": 0.10,
        "top_ignore_ratio": 0.05,
        "border_ignore_ratio": 0.02,
    },
    "classifier": {
        "confidence_floor": 0.55,
        "small_object_recovery_conf": 0.75,
        "high_conf_override": 0.85,
        "immediate_switch_conf": 0.85,
    },
    "tracking": {
        "max_missing_frames": 3,
        "history_size": 5,
        "switch_confirmations": 2,
        "static_hold_conf_decay": 0.85,
        "moving_hold_conf_decay": 0.90,
        "hold_conf_floor": 0.20,
        "moving_threshold_px": 12.0,
    },
    "behavior": {
        "high_speed_px": 35.0,
    },
    "alerts": {
        "enabled": True,
        "enable_sound": True,
        "sound_cooldown_seconds": 2.0,
        "banner_text": "Weapon detected. Immediate response required.",
    },
    "logging": {
        "enabled": True,
        "output_dir": "outputs/events",
        "csv_name": "detections.csv",
        "jsonl_name": "detections.jsonl",
        "snapshots_dir": "outputs/snapshots",
        "snapshot_cooldown_seconds": 2.0,
    },
    "recording": {
        "enabled_by_default": False,
        "output_dir": "outputs/recordings",
        "codec": "mp4v",
    },
    "ui": {
        "max_feeds": 4,
        "grid_columns": 2,
        "camera_sources": [0],
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = Path(config_path) if config_path else root / "config.yaml"

    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return _deep_merge(DEFAULT_CONFIG, loaded)


def resolve_path(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path
