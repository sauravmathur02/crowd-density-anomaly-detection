import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import json
import datetime
import pandas as pd
from collections import defaultdict
from pipelines.main_pipeline import AnomalyDetector, YOLODetector
from utils.video_utils import draw_detections
from utils.risk_engine import calculate_risk_score
from anomaly.optical_flow import MotionAnalyzer

STANDARD_FRAME_WIDTH = 1280
STANDARD_FRAME_HEIGHT = 720


def standardize_frame(frame, target_width=STANDARD_FRAME_WIDTH, target_height=STANDARD_FRAME_HEIGHT):
    """
    Convert every input frame into a fixed 16:9 canvas without stretching it.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    scale = min(target_width / float(w), target_height / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_off = (target_width - new_w) // 2
    y_off = (target_height - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

# Configure Page Layout
st.set_page_config(page_title="SentinelVision Research Console", page_icon="🚨", layout="wide")

st.markdown("""
<style>
.red-alert {
    color: white;
    background-color: #ff4b4b;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}
.warning-alert {
    color: black;
    background-color: #ffa500;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}
.safe-alert {
    color: white;
    background-color: #00cc66;
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}
.dark-panel {
    background-color: #2e2e2e; 
    padding: 15px; 
    border-radius: 10px; 
    margin-bottom: 10px;
}
details > summary { list-style: none; }
details > summary::-webkit-details-marker { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SentinelVision Research Console")

@st.cache_resource
def load_yolo_model(yolo_path):
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Model not found: {yolo_path}")
    return YOLODetector(weights_path=yolo_path, conf=0.08, iou=0.72, imgsz=960)

@st.cache_resource
def load_anomaly_model():
    ae_path = "weights/ae_best.pth"
    thresh_path = "weights/ae_threshold.json"
    
    threshold = 0.01
    if os.path.exists(thresh_path):
        try:

            payload = json.loads(open(thresh_path).read())
            threshold = float(payload.get("threshold_mse", 0.01))
        except: pass
        
    return AnomalyDetector(weights_path=ae_path if os.path.exists(ae_path) else None, threshold=threshold, temporal=True)

st.sidebar.header("⚙️ Configuration Panel")
model_choice = st.sidebar.selectbox(
    "Detection Model", 
    [
        "yolov8m.pt (Default)",
        "weights/best_v2.pt (Old Model)",
        "runs/detect/train5/weights/best.pt (NEW MODEL 🔥)"
    ],
    index=2
)
yolo_path_choice = model_choice.split(" ")[0]

try:
    yolo_model = load_yolo_model(yolo_path_choice)
    ae_model = load_anomaly_model()
except Exception as e:
    st.error(f"Error loading core architectural AI models: {e}")
    st.stop()

input_source = st.sidebar.radio("Video Source", ["Upload Video", "Upload Image", "Webcam (Live)"])
uploaded_file = None

if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov"])
elif input_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
else:
    webcam_id = st.sidebar.number_input("Webcam Hardware ID", min_value=0, max_value=10, value=0)

st.sidebar.subheader("🚨 Security Mode")
security_mode = st.sidebar.radio("Preset Response Level", ["Normal", "Strict", "Emergency"], 
                                help="Emergency heavily prioritizes motion and density into critical risk.")
show_heatmap = st.sidebar.toggle("Overlay Crowd Heatmap", value=True)

st.sidebar.subheader("⚡ Performance")
frame_skip = st.sidebar.slider("Throttle (Process 1 in N frames)", 1, 10, 2)

st.sidebar.markdown("---")

if 'protocol_running' not in st.session_state: st.session_state.protocol_running = False
if 'is_paused' not in st.session_state: st.session_state.is_paused = False

col_b1, col_b2, col_b3 = st.sidebar.columns([1, 1, 1])
start_button = col_b1.button("▶️ Start")
pause_button = col_b2.button("⏸️ Pause")  
stop_button = col_b3.button("⏹️ Stop")

if start_button:
    st.session_state.protocol_running = True
    st.session_state.is_paused = False
    st.session_state.frame_idx = 0
    st.session_state.vid_path = None
    st.session_state.track_history = defaultdict(list)
    st.session_state.motion_analyzer = MotionAnalyzer()
    st.session_state.risk_history = []
    st.session_state.velocity_history = []
    st.session_state.alert_logs = []
    yolo_model.reset_tracking()
    
    if input_source == "Upload Video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        st.session_state.vid_path = tfile.name
    elif input_source == "Webcam (Live)":
        st.session_state.vid_path = int(webcam_id)

if stop_button:
    st.session_state.protocol_running = False
    yolo_model.reset_tracking()
if pause_button:
    st.session_state.is_paused = not st.session_state.is_paused

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Intelligence Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Dynamic Risk Analytics")
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    st.markdown("### 📈 Live Telemetry")
    chart_placeholder_risk = st.empty()
    chart_placeholder_velocity = st.empty()
    
    st.markdown("### 📸 Incident Snapshot Log")
    log_placeholder = st.empty()

if start_button and input_source == "Upload Image":
    if uploaded_file is None:
        st.sidebar.error("Upload an image first!")
        st.stop()
        
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    if frame is None:
        st.error("Failed to decode image input.")
        st.stop()
    frame = standardize_frame(frame)
        
    # ---------------------------- #
    # SINGLE IMAGE ENGINE BYPASS   #
    # ---------------------------- #
    detections = yolo_model.detect(frame, persist=False) 
    people_count = yolo_model.count_class(detections, 0)
    weapon_present = yolo_model.any_class(detections, class_ids=[1, 2])
    
    # Risk for single frame (Anomaly and Optical Flow bypass)
    risk_info = calculate_risk_score(
        people_count=people_count,
        weapon_detected=weapon_present,
        anomaly_mse=0.0,
        anomaly_threshold=1.0,
        crowd_high_thresh=15, 
        optical_flow_mag=0.0,
        mode=security_mode
    )
    
    annotated = draw_detections(
        frame_bgr=frame,
        detections=detections,
        class_id_to_name=yolo_model.class_names,
        risk_info=risk_info,
        track_history=defaultdict(list),
        show_heatmap=show_heatmap
    )
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Status
    score = risk_info["score"]
    severity = risk_info["severity"]
    status_html = f"""
    <div class='dark-panel'>
        <h3 style='margin:0;color:white;'>Image Analysis Status</h3><hr/>
        <strong style='color:white;'>Active Model:</strong> <span style='color: #ffaa00;'>{yolo_model.weights_path}</span><br>
        <strong style='color:white;'>Security Mode:</strong> <span style='color: lightblue;'>{security_mode}</span><br>
        <strong style='color:white;'>Detected Persons:</strong> <span style='font-size: 20px; color: lightblue;'>{people_count}</span><br>
    </div>
    """
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    if severity == "CRITICAL":
        alert_placeholder.markdown("<div class='red-alert'>🚨 CRITICAL THREAT ENVIRONMENT 🚨</div>", unsafe_allow_html=True)
    elif severity in {"HIGH", "WARNING", "ELEVATED"}:
        alert_placeholder.markdown("<div class='warning-alert'>⚠️ ELEVATED WARNING ⚠️</div>", unsafe_allow_html=True)
        
    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
    st.stop()
        
# ------ VIDEO AND WEBCAM LOGIC ------- #
if st.session_state.protocol_running and input_source != "Upload Image":
    video_path = st.session_state.vid_path
    if video_path is None:
        st.error("Failed to upload or recover video source.")
        st.session_state.protocol_running = False
        st.stop()
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to initialize video input.")
        st.stop()
        
    # Accurately jump to saved pause state frame if it's an uploaded video!
    if input_source == "Upload Video" and 'frame_idx' in st.session_state and st.session_state.frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
        
    # State tracking pulled from session state so paused reruns don't erase graphs!
    track_history = st.session_state.track_history
    motion_analyzer = st.session_state.motion_analyzer
    risk_history = st.session_state.risk_history
    velocity_history = st.session_state.velocity_history
    alert_logs = st.session_state.alert_logs
    
    while cap.isOpened() and st.session_state.protocol_running:
        if st.session_state.is_paused:
            if 'last_rendered_frame' in st.session_state:
                video_placeholder.image(st.session_state.last_rendered_frame, channels="RGB", use_container_width=True)
            status_placeholder.markdown("<div class='warning-alert'>⏸️ Video Processing Paused</div>", unsafe_allow_html=True)
            break
            
        ret, frame = cap.read()
        if not ret:
            st.info("End of feed/video stream.")
            st.session_state.protocol_running = False
            break
        frame = standardize_frame(frame)
            
        st.session_state.frame_idx += 1

        if st.session_state.frame_idx % frame_skip != 0:
            continue

        # ---------------------------- #
        # UNIFIED INTELLIGENCE ENGINE  #
        # ---------------------------- #
        
        # 1. ByteTrack Object Tracking (YOLOv8)
        detections = yolo_model.detect(frame, persist=True) 
        people_count = yolo_model.count_class(detections, 0)
        weapon_present = yolo_model.any_class(detections, class_ids=[1, 2])
        
        # 2. Temporal Anomaly Analysis (ConvLSTM)
        mse, is_anomaly = ae_model.score_frame(frame)
        anomaly_threshold = getattr(ae_model, "current_threshold", ae_model.threshold)

        # 3. Optical Flow Motion Speed (Panic/Fight Detection)
        flow_mag = motion_analyzer.analyze_motion(frame, frame_stride=frame_skip)
        
        # 4. Probabilistic Risk Compilation
        risk_info = calculate_risk_score(
            people_count=people_count,
            weapon_detected=weapon_present,
            anomaly_mse=mse,
            anomaly_threshold=anomaly_threshold,
            crowd_high_thresh=15, # Adaptive baseline
            optical_flow_mag=flow_mag,
            mode=security_mode
        )
        
        score = risk_info["score"]
        severity = risk_info["severity"]
        
        # Mandatory Historic Logging Triggers
        trigger_alert = (
            severity in ["HIGH", "WARNING", "CRITICAL"] 
            or people_count >= 15 
            or weapon_present 
            or is_anomaly
        )
        
        # Draw Overlays (Trajectories, Heatmaps, UI Metrics)
        annotated = draw_detections(
            frame_bgr=frame,
            detections=detections,
            class_id_to_name=yolo_model.class_names,
            risk_info=risk_info,
            track_history=track_history,
            show_heatmap=show_heatmap
        )
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # ---------------------------- #
        # UI RENDERING & LOGGING       #
        # ---------------------------- #
        
        if trigger_alert:
            is_spam = False
            
            # Determine explicit cause
            if weapon_present: trigger_msg = "WEAPON_DETECTED"
            elif is_anomaly: trigger_msg = "ANOMALY_BEHAVIOR"
            elif people_count >= 15: trigger_msg = f"HIGH_CROWD_{people_count}"
            else: trigger_msg = f"{severity}_RISK"
            
            if len(alert_logs) > 0:
                last_alert = alert_logs[0]
                if (st.session_state.frame_idx - last_alert["frame_idx"]) < 20:
                    is_spam = True
                    
            if not is_spam:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                safe_msg = f"{score}%_{severity}"
                filename = f"outputs/alert_history/{current_date}_{current_time.replace(':', '-')}_{safe_msg}.jpg"
                
                os.makedirs("outputs/alert_history", exist_ok=True)
                cv2.imwrite(filename, annotated)
                
                alert_logs.insert(0, {
                    "time": current_time,
                    "date": current_date,
                    "msg": trigger_msg,
                    "score": score,
                    "img_rgb": annotated_rgb.copy(),
                    "frame_idx": st.session_state.frame_idx,
                    "saved_path": filename
                })

        if len(alert_logs) > 10:
            del alert_logs[10:]
            
        # Draw Risk Status Panel
        status_html = f"""
        <div class='dark-panel'>
            <h3 style='margin:0;color:white;'>Active Status</h3><hr/>
            <strong style='color:white;'>Active Model:</strong> <span style='color: #ffaa00;'>{yolo_model.weights_path}</span><br>
            <strong style='color:white;'>Security Mode:</strong> <span style='color: lightblue;'>{security_mode}</span><br>
            <strong style='color:white;'>Total Tracked Persons:</strong> <span style='font-size: 20px; color: lightblue;'>{people_count}</span><br>
            <strong style='color:white;'>Average Motion Flow:</strong> <span style='color: lightblue;'>{flow_mag:.2f}px/frame</span><br>
        </div>
        """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)
        
        if severity == "CRITICAL":
            alert_placeholder.markdown("<div class='red-alert'>🚨 CRITICAL THREAT ENVIRONMENT 🚨</div>", unsafe_allow_html=True)
        elif severity in {"HIGH", "WARNING"}:
            alert_placeholder.markdown("<div class='warning-alert'>⚠️ ELEVATED WARNING ⚠️</div>", unsafe_allow_html=True)
        else:
            alert_placeholder.empty()
            
        with log_placeholder.container():
            if not alert_logs:
                st.markdown("<p style='color: gray;'>No major incidents logged.</p>", unsafe_allow_html=True)
            else:
                for log in alert_logs:
                    with st.popover(f"📸 [{log['time']}] {log['msg']} (Risk: {log['score']}%)", use_container_width=True):
                        st.image(log["img_rgb"], channels="RGB", use_container_width=True)
        
        # Telemetry Graphs
        risk_history.append(score)
        velocity_history.append(flow_mag)
        if len(risk_history) > 60:
            risk_history.pop(0)
            velocity_history.pop(0)

        chart_placeholder_risk.line_chart(pd.DataFrame(risk_history, columns=["Risk Score (%)"]), height=150)
        chart_placeholder_velocity.line_chart(pd.DataFrame(velocity_history, columns=["Crowd Velocity"]), height=150)
        
        video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        st.session_state.last_rendered_frame = annotated_rgb
        
    cap.release()
    yolo_model.reset_tracking()
    try:
        # Avoid destroying video_path on pause, only kill it when terminating formally to memory
        if not st.session_state.protocol_running and input_source == "Upload Video" and video_path and os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass
