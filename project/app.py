from __future__ import annotations

import tempfile
import time
from collections import deque
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from backend.alerts import get_beep_audio_html
from backend.pipeline import FeedProcessor, ProcessedFrame, RenderOptions, RuntimeFactory


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class FeedSpec:
    source_id: str
    label: str
    capture_target: object
    temp_path: Optional[Path] = None


@dataclass
class FeedSession:
    spec: FeedSpec
    capture: cv2.VideoCapture
    processor: FeedProcessor


st.set_page_config(page_title="AI Surveillance Command Center", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def load_runtime(config_mtime: float) -> RuntimeFactory:
    return RuntimeFactory(ROOT, CONFIG_PATH)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at top left, rgba(18,50,86,.55), transparent 28%), linear-gradient(180deg,#07111d 0%,#040910 100%); color: #f7fbff; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#0a1422 0%, #08101a 100%); border-right: 1px solid rgba(255,255,255,.08); }
        .hero, .card, .banner { background: rgba(10,20,34,.92); border: 1px solid rgba(148,163,184,.16); border-radius: 18px; box-shadow: 0 18px 40px rgba(0,0,0,.28); }
        .hero { padding: 1.25rem 1.35rem; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(25,192,165,.16), rgba(245,158,11,.10)), rgba(10,20,34,.96); }
        .hero h1 { margin: 0; font-size: 2rem; letter-spacing: .02em; }
        .hero p { margin: .35rem 0 0 0; color: #93a6ba; }
        .banner { padding: .95rem 1.15rem; margin-bottom: 1rem; color: #dcfce7; background: rgba(6,78,59,.34); }
        .banner.alert { color: #fee2e2; border-color: rgba(239,68,68,.42); background: linear-gradient(90deg, rgba(127,29,29,.9), rgba(69,10,10,.88)); animation: pulseAlert 1.15s infinite ease-in-out; }
        .cards4, .cards2 { display: grid; gap: .85rem; }
        .cards4 { grid-template-columns: repeat(4, minmax(0,1fr)); }
        .cards2 { grid-template-columns: repeat(2, minmax(0,1fr)); }
        .card { padding: .95rem 1rem; margin-bottom: 1rem; }
        .label { color: #90a4b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
        .value { margin-top: .25rem; font-size: 1.6rem; font-weight: 800; }
        .sub { margin-top: .18rem; color: #90a4b8; font-size: .84rem; }
        .safe { border-color: rgba(34,197,94,.28); }
        .warn { border-color: rgba(245,158,11,.28); }
        .danger { border-color: rgba(239,68,68,.30); }
        .risk { padding: 1.1rem 1.15rem; }
        .risk .value { font-size: 2rem; }
        .panel-title { margin: 0 0 .75rem 0; font-size: .95rem; font-weight: 700; }
        .pills { display: flex; gap: .5rem; flex-wrap: wrap; }
        .pill { padding: .35rem .7rem; border-radius: 999px; font-size: .82rem; font-weight: 700; background: rgba(148,163,184,.12); border: 1px solid rgba(148,163,184,.18); }
        .pill.gun { color: #fecaca; background: rgba(127,29,29,.34); border-color: rgba(239,68,68,.32); }
        .pill.knife { color: #fed7aa; background: rgba(120,53,15,.32); border-color: rgba(245,158,11,.32); }
        .pill.none { color: #bbf7d0; background: rgba(20,83,45,.28); border-color: rgba(34,197,94,.28); }
        .confidence-row { margin-bottom: .72rem; }
        .confidence-head { display: flex; justify-content: space-between; gap: .7rem; font-size: .88rem; margin-bottom: .3rem; }
        .confidence-bar { height: 10px; border-radius: 999px; background: rgba(148,163,184,.12); overflow: hidden; }
        .confidence-fill { display: block; height: 100%; border-radius: 999px; }
        .event-log { max-height: 420px; overflow-y: auto; }
        .event { display: grid; grid-template-columns: 92px 1fr auto; gap: .75rem; padding: .68rem 0; border-bottom: 1px solid rgba(148,163,184,.12); }
        .muted { color: #90a4b8; font-size: .82rem; }
        .feed-meta { color: #90a4b8; font-size: .84rem; margin: .45rem 0 .1rem 0; }
        .section { margin-bottom: .7rem; font-size: 1rem; font-weight: 700; }
        @keyframes pulseAlert { 0% { box-shadow: 0 0 0 0 rgba(239,68,68,.22); } 50% { box-shadow: 0 0 0 12px rgba(239,68,68,0); } 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); } }
        @media (max-width: 1200px) { .cards4, .cards2 { grid-template-columns: repeat(2, minmax(0,1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_card(title: str, value: str, subtitle: str = "", tone: str = "safe") -> str:
    return (
        f"<div class='card {tone}'>"
        f"<div class='label'>{escape(title)}</div>"
        f"<div class='value'>{escape(value)}</div>"
        f"<div class='sub'>{escape(subtitle)}</div>"
        "</div>"
    )


def banner_html(active: bool, text: str) -> str:
    classes = "banner alert" if active else "banner"
    return f"<div class='{classes}'>{escape(text)}</div>"


def confidence_html(result: Optional[ProcessedFrame]) -> str:
    if result is None or not result.detections:
        return "<div class='card'>Confidence bars appear when detections arrive.</div>"

    colors = {"person": "#22c55e", "gun": "#ef4444", "knife": "#f59e0b", "weapon": "#ef4444"}
    rows = []
    for det in sorted(result.detections, key=lambda item: item.conf, reverse=True)[:6]:
        width = max(4, min(100, int(round(det.conf * 100))))
        color = colors.get(det.display_label, "#19c0a5")
        rows.append(
            "<div class='confidence-row'>"
            f"<div class='confidence-head'><span>{escape(det.display_label.title())}</span><span>{width}%</span></div>"
            f"<div class='confidence-bar'><span class='confidence-fill' style='width:{width}%; background:{color};'></span></div>"
            "</div>"
        )
    return "<div class='card'><div class='panel-title'>Confidence Scores</div>" + "".join(rows) + "</div>"


def event_log_html(entries: Deque[dict]) -> str:
    if not entries:
        return "<div class='card'>Detection events will stream here during monitoring.</div>"

    rows = []
    for item in reversed(entries):
        rows.append(
            "<div class='event'>"
            f"<div class='muted'>{escape(item['time'])}</div>"
            f"<div><div>{escape(item['label'])}</div><div class='muted'>{escape(item['source'])} | {escape(item['meta'])}</div></div>"
            f"<div>{escape(item['confidence'])}</div>"
            "</div>"
        )
    return "<div class='card'><div class='panel-title'>Event Log</div><div class='event-log'>" + "".join(rows) + "</div></div>"


def persist_upload(uploaded_file) -> Path:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()
    return Path(temp_file.name)


def decode_uploaded_image(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    if not image_bytes:
        return None
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return image


def build_feed_specs(source_mode: str, uploaded_files, source_text: str, runtime: RuntimeFactory) -> List[FeedSpec]:
    max_feeds = int(runtime.config.get("ui", {}).get("max_feeds", 4))
    specs: List[FeedSpec] = []
    if source_mode == "Upload Images":
        for idx, uploaded_file in enumerate((uploaded_files or [])[:max_feeds]):
            specs.append(FeedSpec(f"image-{idx + 1}", f"Image {idx + 1}: {uploaded_file.name}", uploaded_file))
        return specs
    if source_mode == "Upload Videos":
        for idx, uploaded_file in enumerate((uploaded_files or [])[:max_feeds]):
            specs.append(FeedSpec(f"upload-{idx + 1}", f"Upload {idx + 1}: {uploaded_file.name}", uploaded_file))
        return specs

    raw_sources = [line.strip() for line in source_text.splitlines() if line.strip()]
    for idx, value in enumerate(raw_sources[:max_feeds]):
        specs.append(FeedSpec(f"camera-{idx}", f"Camera {idx}: {value}", int(value) if value.isdigit() else value))
    return specs


def open_sessions(feed_specs: List[FeedSpec], runtime: RuntimeFactory, enable_recording: bool) -> List[FeedSession]:
    sessions: List[FeedSession] = []
    for spec in feed_specs:
        target = spec.capture_target
        if hasattr(target, "read"):
            spec.temp_path = persist_upload(target)
            target = str(spec.temp_path)
        capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            capture.release()
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        sessions.append(
            FeedSession(
                spec=spec,
                capture=capture,
                processor=runtime.create_feed_processor(spec.source_id, enable_recording=enable_recording, fps=fps, persist=False),
            )
        )
    return sessions


def apply_processor_controls(
    processor: FeedProcessor,
    runtime: RuntimeFactory,
    person_threshold: float,
    weapon_threshold: float,
    classifier_floor: float,
    enable_tracking: bool,
) -> None:
    tracking_defaults = runtime.config.get("tracking", {})
    runtime.classifier.confidence_floor = classifier_floor
    detector = processor.detector
    detector.person_threshold = person_threshold
    detector.min_person_yolo_conf = person_threshold
    detector.weapon_threshold = max(0.15, weapon_threshold - 0.02)
    detector.base_weapon_yolo_conf = weapon_threshold
    detector.low_weapon_yolo_conf = max(0.15, weapon_threshold - 0.05)
    detector.min_weapon_candidate_conf = max(0.10, weapon_threshold - 0.08)
    smoother = processor.smoother
    if enable_tracking:
        smoother.max_missing_frames = int(tracking_defaults.get("max_missing_frames", 3))
        smoother.switch_confirmations = int(tracking_defaults.get("switch_confirmations", 2))
    else:
        smoother.max_missing_frames = 0
        smoother.switch_confirmations = 1
        processor.track_history.clear()


def apply_controls(
    sessions: List[FeedSession],
    runtime: RuntimeFactory,
    person_threshold: float,
    weapon_threshold: float,
    classifier_floor: float,
    enable_tracking: bool,
) -> None:
    for session in sessions:
        apply_processor_controls(
            session.processor,
            runtime,
            person_threshold,
            weapon_threshold,
            classifier_floor,
            enable_tracking,
        )


def render_preview_slots(feed_specs: List[FeedSpec], grid_columns: int):
    slots = {}
    if len(feed_specs) <= 1:
        return slots
    for row_start in range(0, len(feed_specs), grid_columns):
        row_specs = feed_specs[row_start:row_start + grid_columns]
        cols = st.columns(len(row_specs))
        for spec, col in zip(row_specs, cols):
            with col:
                st.markdown(f"<div class='panel-title'>{escape(spec.label)}</div>", unsafe_allow_html=True)
                slots[spec.source_id] = {"image": st.empty(), "meta": st.empty()}
    return slots


def update_event_log(entries: Deque[dict], seen: Dict[tuple, float], result: ProcessedFrame, now_ts: float) -> None:
    ranked = sorted(result.detections, key=lambda item: (0 if item.label == "weapon" else 1, -item.conf))[:5]
    for det in ranked:
        key = (result.source_id, det.track_id, det.display_label)
        if now_ts - seen.get(key, 0.0) < 4.0:
            continue
        seen[key] = now_ts
        entries.append(
            {
                "time": result.timestamp.split("T")[-1],
                "label": det.display_label.upper(),
                "source": result.source_id,
                "meta": f"Track {det.track_id if det.track_id is not None else 'NA'}",
                "confidence": f"{int(round(det.conf * 100))}%",
            }
        )


def update_fps(fps_state: Dict[str, Dict[str, float]], source_id: str, now_ts: float) -> float:
    state = fps_state.get(source_id)
    if state is None:
        fps_state[source_id] = {"last": now_ts, "fps": 0.0}
        return 0.0
    delta = max(1e-6, now_ts - state["last"])
    instant = 1.0 / delta
    state["fps"] = instant if state["fps"] == 0.0 else (state["fps"] * 0.75) + (instant * 0.25)
    state["last"] = now_ts
    return state["fps"]


inject_styles()
st.markdown(
    "<div class='hero'><h1>AI Surveillance Command Center</h1><p>Modern live dashboard for annotated video, crowd intelligence, weapon alerts, confidence tracking, and event awareness.</p></div>",
    unsafe_allow_html=True,
)

config_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
runtime = load_runtime(config_mtime)
config = runtime.config
if not runtime.classifier.available:
    st.error(f"Classifier checkpoint is not available at {runtime.classifier_weights}.")
    st.stop()

st.sidebar.header("Control Panel")
source_mode = st.sidebar.radio("Video Source", ["Upload Videos", "Upload Images", "Camera / Stream URLs"])
uploaded_files = []
source_text = ""
if source_mode == "Upload Videos":
    uploaded_files = st.sidebar.file_uploader("Upload one or more videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)
elif source_mode == "Upload Images":
    uploaded_files = st.sidebar.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
else:
    default_sources = "\n".join(str(value) for value in config.get("ui", {}).get("camera_sources", [0]))
    source_text = st.sidebar.text_area("Camera indexes or RTSP/HTTP URLs", value=default_sources, height=120)

st.sidebar.subheader("Confidence Controls")
person_threshold = st.sidebar.slider("Person Threshold", 0.10, 0.80, float(config.get("detection", {}).get("person_threshold", 0.20)), 0.01)
weapon_threshold = st.sidebar.slider("Weapon YOLO Threshold", 0.10, 0.80, float(config.get("detection", {}).get("base_weapon_yolo_conf", 0.20)), 0.01)
classifier_floor = st.sidebar.slider("Classifier Floor", 0.30, 0.95, float(runtime.classifier.confidence_floor), 0.01)

st.sidebar.subheader("Overlay Controls")
show_boxes = st.sidebar.toggle("Show Boxes", value=True)
show_labels = st.sidebar.toggle("Show Labels", value=True)
enable_tracking = st.sidebar.toggle("Enable Tracking", value=True)
enable_recording = st.sidebar.checkbox("Record Annotated Output", value=bool(config.get("recording", {}).get("enabled_by_default", False)))

feed_specs = build_feed_specs(source_mode, uploaded_files, source_text, runtime)
focus_options = {spec.label: spec.source_id for spec in feed_specs}
focus_label = st.sidebar.selectbox("Focused Feed", list(focus_options.keys()), disabled=not focus_options) if focus_options else None
start = st.sidebar.button("Launch Dashboard", type="primary")

render_options = RenderOptions(show_boxes=show_boxes, show_labels=show_labels, show_tracking=enable_tracking, highlight_objects=True)

banner_placeholder = st.empty()
top_cards_placeholder = st.empty()
left_col, right_col = st.columns([1.6, 1.0], gap="large")
with left_col:
    st.markdown("<div class='section'>Live Video Feed</div>", unsafe_allow_html=True)
    focus_meta_placeholder = st.empty()
    focus_frame_placeholder = st.empty()
    if len(feed_specs) > 1:
        st.markdown("<div class='section'>Additional Feeds</div>", unsafe_allow_html=True)
    preview_slots = render_preview_slots(feed_specs, int(config.get("ui", {}).get("grid_columns", 2))) if feed_specs else {}
with right_col:
    risk_placeholder = st.empty()
    quick_metrics_placeholder = st.empty()
    weapon_status_placeholder = st.empty()
    confidence_placeholder = st.empty()
    snapshot_title_placeholder = st.empty()
    snapshot_image_placeholder = st.empty()
    snapshot_caption_placeholder = st.empty()

chart_col, log_col = st.columns([1.15, 0.95], gap="large")
with chart_col:
    chart_title_placeholder = st.empty()
    chart_placeholder = st.empty()
with log_col:
    log_placeholder = st.empty()

if start:
    if not feed_specs:
        st.error("Add at least one image, video file, or camera source.")
        st.stop()

    if source_mode == "Upload Images":
        latest_results: Dict[str, ProcessedFrame] = {}
        event_entries: Deque[dict] = deque(maxlen=140)
        event_seen: Dict[tuple, float] = {}
        chart_history = {"detections": deque(maxlen=80), "weapons": deque(maxlen=80), "tracks": deque(maxlen=80)}
        latest_snapshot_path: Optional[str] = None
        alert_sources: List[str] = []
        processed_count = 0

        for spec in feed_specs:
            frame = decode_uploaded_image(spec.capture_target)
            if frame is None:
                continue

            processor = runtime.create_feed_processor(spec.source_id, enable_recording=False, fps=1.0, persist=False)
            try:
                apply_processor_controls(
                    processor,
                    runtime,
                    person_threshold,
                    weapon_threshold,
                    classifier_floor,
                    enable_tracking,
                )
                result = processor.process_frame(frame, render_options=render_options)
            finally:
                processor.close()

            latest_results[spec.source_id] = result
            processed_count += 1
            now_ts = time.time()
            update_event_log(event_entries, event_seen, result, now_ts)
            chart_history["detections"].append(result.detection_count)
            chart_history["weapons"].append(len(result.weapon_labels))
            chart_history["tracks"].append(result.active_tracks)

            if result.snapshot_path:
                latest_snapshot_path = result.snapshot_path
            if result.alert_active:
                alert_sources.append(spec.label)

            slot = preview_slots.get(spec.source_id)
            if slot:
                slot["image"].image(result.annotated_frame, channels="BGR", width="stretch")
                slot["meta"].markdown(
                    f"<div class='feed-meta'>Risk {escape(result.risk['level'])} | "
                    f"Detections {result.detection_count} | Crowd {result.crowd_count}</div>",
                    unsafe_allow_html=True,
                )

        if not latest_results:
            st.error("Unable to decode any uploaded images.")
            st.stop()

        total_crowd = sum(item.crowd_count for item in latest_results.values())
        total_weapons = sum(len(item.weapon_labels) for item in latest_results.values())
        total_tracks = sum(item.active_tracks for item in latest_results.values())
        top_cards_placeholder.markdown(
            "<div class='cards4'>"
            + html_card("Active Feeds", str(processed_count), "Images analyzed", "safe")
            + html_card("Crowd Count", str(total_crowd), "Across uploaded images", "safe")
            + html_card("Weapon Alerts", str(total_weapons), "Image detections", "danger" if total_weapons else "safe")
            + html_card("Avg FPS", "N/A", f"Tracks {total_tracks}", "warn")
            + "</div>",
            unsafe_allow_html=True,
        )

        if alert_sources:
            banner_placeholder.markdown(banner_html(True, f"Weapon alert active on: {', '.join(alert_sources)}"), unsafe_allow_html=True)
            if bool(config.get("alerts", {}).get("enable_sound", True)):
                st.markdown(get_beep_audio_html(), unsafe_allow_html=True)
        else:
            banner_placeholder.markdown(banner_html(False, "Image analysis complete. No weapon alert is active."), unsafe_allow_html=True)

        focused_result = latest_results.get(focus_options.get(focus_label, "")) if focus_label else next(iter(latest_results.values()), None)
        if focused_result is not None:
            focus_meta_placeholder.markdown(
                f"<div class='feed-meta'>Focused source: {escape(focused_result.source_id)} | Mode: Still image analysis</div>",
                unsafe_allow_html=True,
            )
            focus_frame_placeholder.image(focused_result.annotated_frame, channels="BGR", width="stretch")

            tone = "danger" if focused_result.risk["level"] == "DANGER" else "warn" if focused_result.risk["level"] == "ALERT" else "safe"
            risk_placeholder.markdown(
                f"<div class='card risk {tone}'><div class='label'>Focused Feed Risk</div><div class='value'>{escape(focused_result.risk['level'])}</div><div class='sub'>Score {focused_result.risk['score']} | Mode Still Image | Source {escape(focused_result.source_id)}</div></div>",
                unsafe_allow_html=True,
            )

            weapon_text = ", ".join(sorted(set(focused_result.weapon_labels))) if focused_result.weapon_labels else "NONE"
            quick_metrics_placeholder.markdown(
                "<div class='cards2'>"
                + html_card("Crowd Count", str(focused_result.crowd_count), "Unique tracked people", "safe")
                + html_card("Detection Count", str(focused_result.detection_count), "Objects on image", tone)
                + html_card("Active Tracks", str(focused_result.active_tracks), "Track IDs", "warn")
                + html_card("Weapon Status", weapon_text, "Focused image summary", "danger" if focused_result.weapon_labels else "safe")
                + "</div>",
                unsafe_allow_html=True,
            )

            if focused_result.weapon_labels:
                pills = "".join(
                    f"<span class='pill {escape(label.lower())}'>{escape(label.upper())}</span>"
                    for label in sorted(set(focused_result.weapon_labels))
                )
            else:
                pills = "<span class='pill none'>NONE</span>"
            weapon_status_placeholder.markdown(
                f"<div class='card'><div class='panel-title'>Weapon Status</div><div class='pills'>{pills}</div><div class='sub'>{'Snapshot saved' if focused_result.snapshot_path else 'No alert snapshot yet'}</div></div>",
                unsafe_allow_html=True,
            )
            confidence_placeholder.markdown(confidence_html(focused_result), unsafe_allow_html=True)

        snapshot_title_placeholder.markdown("<div class='panel-title'>Snapshot Preview</div>", unsafe_allow_html=True)
        if latest_snapshot_path and Path(latest_snapshot_path).exists():
            snapshot_image_placeholder.image(str(latest_snapshot_path), width="stretch")
            snapshot_caption_placeholder.markdown(
                f"<div class='feed-meta'>{escape(Path(latest_snapshot_path).name)}</div>",
                unsafe_allow_html=True,
            )
        else:
            snapshot_caption_placeholder.markdown("<div class='card'>Snapshots appear here when a weapon alert is captured.</div>", unsafe_allow_html=True)

        chart_title_placeholder.markdown("<div class='panel-title'>Detection Trend</div>", unsafe_allow_html=True)
        chart_placeholder.line_chart(
            {
                "Detections": list(chart_history["detections"]) or [0],
                "Weapons": list(chart_history["weapons"]) or [0],
                "Tracks": list(chart_history["tracks"]) or [0],
            },
            width="stretch",
        )
        log_placeholder.markdown(event_log_html(event_entries), unsafe_allow_html=True)
    else:
        sessions = open_sessions(feed_specs, runtime, enable_recording)
        if not sessions:
            st.error("Unable to open any requested video sources.")
            st.stop()

        apply_controls(sessions, runtime, person_threshold, weapon_threshold, classifier_floor, enable_tracking)
        fps_state: Dict[str, Dict[str, float]] = {}
        latest_results: Dict[str, ProcessedFrame] = {}
        event_entries: Deque[dict] = deque(maxlen=140)
        event_seen: Dict[tuple, float] = {}
        chart_history = {"detections": deque(maxlen=80), "weapons": deque(maxlen=80), "tracks": deque(maxlen=80)}
        latest_snapshot_path: Optional[str] = None
        sound_cfg = config.get("alerts", {})
        last_beep_at = 0.0
        sound_placeholder = st.empty()

        try:
            while sessions:
                active_sessions: List[FeedSession] = []
                alert_sources: List[str] = []
                total_detections = 0
                total_weapons = 0
                total_tracks = 0

                for session in sessions:
                    ok, frame = session.capture.read()
                    if not ok:
                        session.capture.release()
                        session.processor.close()
                        latest_results.pop(session.spec.source_id, None)
                        continue

                    result = session.processor.process_frame(frame, render_options=render_options)
                    latest_results[session.spec.source_id] = result
                    active_sessions.append(session)

                    now_ts = time.time()
                    fps_value = update_fps(fps_state, session.spec.source_id, now_ts)
                    update_event_log(event_entries, event_seen, result, now_ts)

                    total_detections += result.detection_count
                    total_weapons += len(result.weapon_labels)
                    total_tracks += result.active_tracks
                    if result.snapshot_path:
                        latest_snapshot_path = result.snapshot_path
                    if result.alert_active:
                        alert_sources.append(session.spec.label)

                    slot = preview_slots.get(session.spec.source_id)
                    if slot:
                        slot["image"].image(result.annotated_frame, channels="BGR", width="stretch")
                        slot["meta"].markdown(
                            f"<div class='feed-meta'>Risk {escape(result.risk['level'])} | FPS {fps_value:.1f} | Detections {result.detection_count}</div>",
                            unsafe_allow_html=True,
                        )

                sessions = active_sessions
                if not sessions:
                    break

                chart_history["detections"].append(total_detections)
                chart_history["weapons"].append(total_weapons)
                chart_history["tracks"].append(total_tracks)
                fps_values = [state["fps"] for state in fps_state.values() if state["fps"] > 0]
                avg_fps = f"{(sum(fps_values) / len(fps_values)):.1f}" if fps_values else "0.0"

                top_cards_placeholder.markdown(
                    "<div class='cards4'>"
                    + html_card("Active Feeds", str(len(latest_results)), "Monitored now", "safe")
                    + html_card("Crowd Count", str(sum(item.crowd_count for item in latest_results.values())), "People across feeds", "safe")
                    + html_card("Weapon Alerts", str(total_weapons), "Live weapon detections", "danger" if total_weapons else "safe")
                    + html_card("Avg FPS", avg_fps, f"Tracks {total_tracks}", "warn")
                    + "</div>",
                    unsafe_allow_html=True,
                )

                if alert_sources and bool(sound_cfg.get("enabled", True)):
                    banner_placeholder.markdown(banner_html(True, f"Weapon alert active on: {', '.join(alert_sources)}"), unsafe_allow_html=True)
                    now_ts = time.time()
                    if bool(sound_cfg.get("enable_sound", True)) and now_ts - last_beep_at >= float(sound_cfg.get("sound_cooldown_seconds", 2.0)):
                        sound_placeholder.markdown(f"{get_beep_audio_html()}<!-- {now_ts} -->", unsafe_allow_html=True)
                        last_beep_at = now_ts
                else:
                    banner_placeholder.markdown(banner_html(False, "System online. No weapon alert is active."), unsafe_allow_html=True)
                    sound_placeholder.empty()

                focused_result = latest_results.get(focus_options.get(focus_label, "")) if focus_label else next(iter(latest_results.values()), None)
                focused_fps = fps_state.get(focused_result.source_id, {}).get("fps", 0.0) if focused_result else 0.0

                if focused_result is not None:
                    focus_meta_placeholder.markdown(
                        f"<div class='feed-meta'>Focused source: {escape(focused_result.source_id)} | Recording: {escape(Path(focused_result.recording_path).name) if focused_result.recording_path else 'Off'}</div>",
                        unsafe_allow_html=True,
                    )
                    focus_frame_placeholder.image(focused_result.annotated_frame, channels="BGR", width="stretch")

                    tone = "danger" if focused_result.risk["level"] == "DANGER" else "warn" if focused_result.risk["level"] == "ALERT" else "safe"
                    risk_placeholder.markdown(
                        f"<div class='card risk {tone}'><div class='label'>Focused Feed Risk</div><div class='value'>{escape(focused_result.risk['level'])}</div><div class='sub'>Score {focused_result.risk['score']} | FPS {focused_fps:.1f} | Source {escape(focused_result.source_id)}</div></div>",
                        unsafe_allow_html=True,
                    )

                    weapon_text = ", ".join(sorted(set(focused_result.weapon_labels))) if focused_result.weapon_labels else "NONE"
                    quick_metrics_placeholder.markdown(
                        "<div class='cards2'>"
                        + html_card("Crowd Count", str(focused_result.crowd_count), "Unique tracked people", "safe")
                        + html_card("Detection Count", str(focused_result.detection_count), "Objects on frame", tone)
                        + html_card("Active Tracks", str(focused_result.active_tracks), "Track IDs", "warn")
                        + html_card("Weapon Status", weapon_text, "Focused feed summary", "danger" if focused_result.weapon_labels else "safe")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    if focused_result.weapon_labels:
                        pills = "".join(
                            f"<span class='pill {escape(label.lower())}'>{escape(label.upper())}</span>"
                            for label in sorted(set(focused_result.weapon_labels))
                        )
                    else:
                        pills = "<span class='pill none'>NONE</span>"
                    weapon_status_placeholder.markdown(
                        f"<div class='card'><div class='panel-title'>Weapon Status</div><div class='pills'>{pills}</div><div class='sub'>{'Snapshot saved' if focused_result.snapshot_path else 'No alert snapshot yet'}</div></div>",
                        unsafe_allow_html=True,
                    )
                    confidence_placeholder.markdown(confidence_html(focused_result), unsafe_allow_html=True)
                else:
                    risk_placeholder.markdown("<div class='card risk'><div class='label'>Focused Feed Risk</div><div class='value'>IDLE</div><div class='sub'>Waiting for a live source</div></div>", unsafe_allow_html=True)
                    quick_metrics_placeholder.markdown("<div class='card'>Focused feed metrics will appear here.</div>", unsafe_allow_html=True)
                    weapon_status_placeholder.markdown("<div class='card'>Weapon status will update with the live feed.</div>", unsafe_allow_html=True)
                    confidence_placeholder.markdown(confidence_html(None), unsafe_allow_html=True)

                snapshot_title_placeholder.markdown("<div class='panel-title'>Snapshot Preview</div>", unsafe_allow_html=True)
                if latest_snapshot_path and Path(latest_snapshot_path).exists():
                    snapshot_image_placeholder.image(str(latest_snapshot_path), width="stretch")
                    snapshot_caption_placeholder.markdown(f"<div class='feed-meta'>{escape(Path(latest_snapshot_path).name)}</div>", unsafe_allow_html=True)
                else:
                    snapshot_image_placeholder.empty()
                    snapshot_caption_placeholder.markdown("<div class='card'>Snapshots appear here when a weapon alert is captured.</div>", unsafe_allow_html=True)

                chart_title_placeholder.markdown("<div class='panel-title'>Detection Trend</div>", unsafe_allow_html=True)
                chart_placeholder.line_chart(
                    {
                        "Detections": list(chart_history["detections"]),
                        "Weapons": list(chart_history["weapons"]),
                        "Tracks": list(chart_history["tracks"]),
                    },
                    width="stretch",
                )
                log_placeholder.markdown(event_log_html(event_entries), unsafe_allow_html=True)
        finally:
            for session in sessions:
                session.capture.release()
                session.processor.close()
            for spec in feed_specs:
                if spec.temp_path:
                    try:
                        spec.temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
else:
    banner_placeholder.markdown(banner_html(False, "Load a source and launch the dashboard to begin monitoring."), unsafe_allow_html=True)
    top_cards_placeholder.markdown(
        "<div class='cards4'>"
        + html_card("Active Feeds", "0", "Waiting for input", "safe")
        + html_card("Crowd Count", "0", "No live session", "safe")
        + html_card("Weapon Alerts", "0", "No detections yet", "danger")
        + html_card("Avg FPS", "0.0", "Idle", "warn")
        + "</div>",
        unsafe_allow_html=True,
    )
    focus_meta_placeholder.markdown("<div class='feed-meta'>The focused live feed will appear here once monitoring begins.</div>", unsafe_allow_html=True)
    risk_placeholder.markdown("<div class='card risk'><div class='label'>Focused Feed Risk</div><div class='value'>IDLE</div><div class='sub'>Waiting for a live source</div></div>", unsafe_allow_html=True)
    quick_metrics_placeholder.markdown("<div class='card'>Real-time metrics will populate here.</div>", unsafe_allow_html=True)
    weapon_status_placeholder.markdown("<div class='card'>Weapon status will update with the focused feed.</div>", unsafe_allow_html=True)
    confidence_placeholder.markdown(confidence_html(None), unsafe_allow_html=True)
    snapshot_title_placeholder.markdown("<div class='panel-title'>Snapshot Preview</div>", unsafe_allow_html=True)
    snapshot_caption_placeholder.markdown("<div class='card'>Alert snapshots will appear here.</div>", unsafe_allow_html=True)
    chart_title_placeholder.markdown("<div class='panel-title'>Detection Trend</div>", unsafe_allow_html=True)
    chart_placeholder.line_chart({"Detections": [0], "Weapons": [0], "Tracks": [0]}, width="stretch")
    log_placeholder.markdown(event_log_html(deque()), unsafe_allow_html=True)
