def calculate_risk_score(
    people_count: int,
    weapon_detected: bool,
    anomaly_mse: float,
    anomaly_threshold: float,
    crowd_high_thresh: int,
    optical_flow_mag: float = 0.0,
    mode: str = "Normal",
) -> dict:
    """
    Compute an explainable 0-100 risk score.

    Design goals:
    - Any weapon must never end up SAFE/ELEVATED.
    - Crowd size should matter, but only as a minor contributor.
    - Motion and anomaly scores should increase risk consistently.
    """
    multipliers = {
        "Normal": 1.0,
        "Strict": 1.15,
        "Emergency": 1.35,
    }
    mode_multiplier = multipliers.get(mode, 1.0)

    normalized_anomaly = 0.0
    if anomaly_threshold > 0:
        normalized_anomaly = anomaly_mse / anomaly_threshold

    crowd_component = min(max(people_count, 0) / max(crowd_high_thresh, 1), 1.5) * 8.0
    anomaly_component = min(max(normalized_anomaly, 0.0), 3.0) * 12.0
    motion_component = 0.0
    if optical_flow_mag > 1.5:
        motion_component = min((optical_flow_mag - 1.5) * 4.0, 18.0)

    reasons = []
    base_score = crowd_component + anomaly_component + motion_component
    if people_count >= crowd_high_thresh:
        reasons.append("high_crowd_density")
    if normalized_anomaly >= 1.0:
        reasons.append("behavioral_anomaly")
    if optical_flow_mag > 3.0:
        reasons.append("high_motion")

    if weapon_detected:
        reasons.append("weapon_detected")
        base_score = max(base_score + 70.0, 70.0)

    final_score = min(base_score * mode_multiplier, 100.0)

    if weapon_detected and (normalized_anomaly >= 1.0 or optical_flow_mag > 3.0):
        severity = "CRITICAL"
        color = "#ff4b4b"
    elif weapon_detected:
        severity = "HIGH"
        color = "#ff8c00"
    elif final_score >= 55.0:
        severity = "WARNING"
        color = "#ffa500"
    elif final_score >= 20.0:
        severity = "ELEVATED"
        color = "#00cc66"
    else:
        severity = "SAFE"
        color = "#00cc66"

    return {
        "score": round(final_score, 1),
        "severity": severity,
        "color": color,
        "weapon_detected": weapon_detected,
        "people_count": int(people_count),
        "anomaly_mse": float(anomaly_mse),
        "anomaly_threshold": float(anomaly_threshold),
        "optical_flow_mag": float(optical_flow_mag),
        "crowd_high_thresh": int(crowd_high_thresh),
        "mode": mode,
        "reasons": reasons,
        "components": {
            "crowd": round(crowd_component, 2),
            "anomaly": round(anomaly_component, 2),
            "motion": round(motion_component, 2),
            "weapon_floor_applied": bool(weapon_detected),
        },
    }
