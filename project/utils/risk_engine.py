from __future__ import annotations

from typing import Iterable


WEAPON_SCORES = {
    "gun": 100,
    "knife": 80,
}


def compute_risk(crowd_count: int, weapon_labels: Iterable[str], suspicious: bool) -> dict:
    weapon_labels = list(weapon_labels)
    score = int(crowd_count) * 5

    for label in weapon_labels:
        score += WEAPON_SCORES.get(label, 0)

    if suspicious:
        score += 20

    if score >= 100:
        level = "DANGER"
    elif score >= 40:
        level = "ALERT"
    else:
        level = "SAFE"

    return {
        "score": score,
        "level": level,
        "weapon_count": len(weapon_labels),
        "suspicious": suspicious,
    }
