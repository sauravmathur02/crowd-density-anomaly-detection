from fastapi import FastAPI, UploadFile, File
import cv2
import tempfile
import os
import json
from pipelines.main_pipeline import AnomalyDetector, YOLODetector

app = FastAPI(title="Crowd & Anomaly API")

YOLO_WEIGHTS = "weights/best.pt"
AE_WEIGHTS = "weights/ae_best.pth"
THRESH_JSON = "weights/ae_threshold.json"

yolo = None
ae = None

@app.on_event("startup")
def load_models():
    """Load model architectures exactly once at server startup."""
    global yolo, ae
    yolo = YOLODetector(weights_path=YOLO_WEIGHTS)
    threshold = 0.01
    if os.path.exists(THRESH_JSON):
        payload = json.loads(open(THRESH_JSON).read())
        threshold = float(payload.get("threshold_mse", 0.01))
    ae = AnomalyDetector(weights_path=AE_WEIGHTS, threshold=threshold)

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), crowd_threshold: int = 15):
    """
    Process an uploaded video through the pipeline and return timestamps 
    of when anomalies or weapons or high crowd density are detected.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(await file.read())
        video_path = tfile.name

    yolo.reset_tracking()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30.0
        
    results = {
        "timestamps_with_weapons": [],
        "timestamps_with_anomaly": [],
        "timestamps_high_crowd": [],
        "total_frames": 0,
        "fps": fps
    }
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Pipeline execution
        detections = yolo.detect(frame, persist=True)
        people_count = yolo.count_class(detections, 0)
        weapon_present = yolo.any_class(detections, class_ids=[1, 2])
        mse, anomaly_flag = ae.score_frame(frame)
        
        timestamp = round(frame_idx / fps, 2)
        
        if weapon_present:
            results["timestamps_with_weapons"].append(timestamp)
        if anomaly_flag:
            results["timestamps_with_anomaly"].append(timestamp)
        if people_count >= crowd_threshold:
            results["timestamps_high_crowd"].append(timestamp)
            
        frame_idx += 1
        
    cap.release()
    yolo.reset_tracking()
    os.remove(video_path)
    results["total_frames"] = frame_idx
    
    # Deduplicate timestamps slightly to avoid bloated JSON
    results["timestamps_with_weapons"] = sorted(list(set(results["timestamps_with_weapons"])))
    results["timestamps_with_anomaly"] = sorted(list(set(results["timestamps_with_anomaly"])))
    results["timestamps_high_crowd"] = sorted(list(set(results["timestamps_high_crowd"])))
    
    return results
