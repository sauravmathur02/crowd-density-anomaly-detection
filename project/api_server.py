from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend.pipeline import RuntimeFactory


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
runtime = RuntimeFactory(ROOT, CONFIG_PATH)

app = FastAPI(title="SentinelVision Surveillance API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "detector_weights": str(runtime.detector_weights),
        "classifier_available": runtime.classifier.available,
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    source_id: str = Form("api-camera-0"),
) -> dict:
    if not runtime.classifier.available:
        raise HTTPException(status_code=500, detail="Classifier checkpoint is not available.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image.")

    processor = runtime.get_or_create_processor(source_id=source_id, enable_recording=False, fps=20.0)
    result = processor.process_frame(frame)
    return result.to_response()
