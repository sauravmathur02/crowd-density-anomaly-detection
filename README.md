# AI-Based Crowd Monitoring and Weapon Detection System

Production-oriented real-time surveillance system for crowd monitoring, weapon detection, alerting, event logging, and dashboard-driven review.

## Overview

- Real-time person and weapon monitoring using YOLOv8
- Two-stage weapon reasoning:
  - Stage 1: detect `person` and `weapon`
  - Stage 2: classify weapon ROI as `gun` or `knife` using MobileNetV3
- Temporal smoothing and track-aware stability logic for reduced flicker
- Streamlit dashboard for live monitoring
- FastAPI backend for programmatic `/detect` inference
- Event logging, alert snapshots, and annotated recording support

## Main Application

The production app lives in [`project/`](./project).

- [`project/app.py`](./project/app.py): Streamlit dashboard
- [`project/api_server.py`](./project/api_server.py): FastAPI service
- [`project/backend/pipeline.py`](./project/backend/pipeline.py): shared end-to-end runtime
- [`project/detector/yolo_detector.py`](./project/detector/yolo_detector.py): YOLOv8 detection + tracking-aware filtering
- [`project/classifier/weapon_classifier.py`](./project/classifier/weapon_classifier.py): MobileNetV3 weapon classifier
- [`project/utils/smoothing.py`](./project/utils/smoothing.py): temporal stabilization
- [`project/utils/behavior.py`](./project/utils/behavior.py): motion and suspicious-behavior heuristics
- [`project/utils/risk_engine.py`](./project/utils/risk_engine.py): risk scoring
- [`project/config.yaml`](./project/config.yaml): runtime thresholds and feature toggles

## Key Features

- Real-time annotated video monitoring
- Person detection with tracking-based crowd count
- Weapon detection with gun-vs-knife classification
- Risk level generation
- Alert banner and sound trigger
- Event logging to CSV and JSONL
- Snapshot capture when weapon alerts occur
- Annotated output recording
- Multi-feed dashboard support
- Image upload, video upload, and live camera input

## System Pipeline

```text
Video / Image Input
  -> YOLOv8 Detection
  -> Tracking + Post-processing
  -> Weapon ROI Cropping
  -> MobileNetV3 Gun/Knife Classification
  -> Temporal Smoothing + Behavior Logic
  -> Risk Engine
  -> Streamlit Dashboard / FastAPI Response / Logging
```

## Quick Start

### 1. Install dependencies

```bash
cd project
pip install -r requirements.txt
```

### 2. Add model files

Place model files here:

- `project/models/best.pt`
- `project/models/classifier.pth`

These are intentionally not committed to Git because of size and artifact management.

### 3. Run the dashboard

```bash
cd project
streamlit run app.py
```

### 4. Run the API

```bash
cd project
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## API

### Health

```http
GET /health
```

### Detect

```http
POST /detect
```

Form fields:

- `file`: image file
- `source_id`: optional source identifier

Returns:

- detections
- risk summary
- crowd count
- alert state
- active tracks
- dominant weapon

## Configuration

Runtime behavior is controlled through [`project/config.yaml`](./project/config.yaml).

Configurable areas include:

- detection thresholds
- classifier confidence floors
- temporal smoothing settings
- motion behavior thresholds
- alert behavior
- logging output paths
- recording settings
- UI defaults
- API host and port

## Training and Dataset Utilities

This repository also contains supporting scripts for data preparation and experimentation:

- detector training utilities in [`pipelines/`](./pipelines)
- dataset merge / cleanup scripts in the repo root
- legacy and experimental modules in folders such as [`detection/`](./detection), [`anomaly/`](./anomaly), and [`utils/`](./utils)

Examples:

- [`merge_dataset.py`](./merge_dataset.py)
- [`merge_yolo_datasets_strict.py`](./merge_yolo_datasets_strict.py)
- [`coco_to_yolo_person.py`](./coco_to_yolo_person.py)
- [`split_coco_person.py`](./split_coco_person.py)
- [`pipelines/train_yolo_detector.py`](./pipelines/train_yolo_detector.py)
- [`project/classifier/build_weapon_dataset.py`](./project/classifier/build_weapon_dataset.py)
- [`project/classifier/train_weapon_classifier.py`](./project/classifier/train_weapon_classifier.py)

## Repository Notes

- Large datasets, videos, model weights, and outputs are excluded through [`.gitignore`](./.gitignore)
- The recommended entry point for demos and deployment is the `project/` application
- Root-level scripts are retained because they document the full engineering workflow: preprocessing, merging, training, evaluation, and prototyping

## Suggested Git Workflow

```bash
git checkout -b feature/ai-surveillance-system
git add .
git commit -m "Add AI surveillance system pipeline, dashboard, and dataset utilities"
git push -u origin feature/ai-surveillance-system
```

## Practical Use Cases

- smart city surveillance
- campus security monitoring
- mall and station monitoring
- restricted-zone weapon alerting
- incident review and event logging

