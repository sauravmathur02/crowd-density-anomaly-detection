import argparse
import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score
from anomaly.anomaly_detector import AnomalyDetector
from utils.video_utils import get_capture

def evaluate_yolo(weights_path, data_yaml):
    print(f"Evaluating YOLO model: {weights_path} on dataset: {data_yaml}")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    print("\n--- YOLO Evaluation Complete ---")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

def get_video_max_mse(video_path, ae_detector):
    cap = get_capture(video_path)
    max_mse = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mse, _ = ae_detector.score_frame(frame)
        if mse > max_mse:
            max_mse = mse
    cap.release()
    return max_mse

def evaluate_anomaly(ae_weights, normal_dir, anomaly_dir):
    print(f"Evaluating ConvAE model: {ae_weights}")
    ae_detector = AnomalyDetector(weights_path=ae_weights)
    
    y_true = []
    y_scores = []
    
    # Process Normal videos (Label = 0)
    for file in os.listdir(normal_dir):
        if file.endswith((".mp4", ".avi")):
            print(f"Processing normal video: {file}")
            path = os.path.join(normal_dir, file)
            score = get_video_max_mse(path, ae_detector)
            y_true.append(0)
            y_scores.append(score)
            
    # Process Anomalous videos (Label = 1)
    for file in os.listdir(anomaly_dir):
        if file.endswith((".mp4", ".avi")):
            print(f"Processing anomalous video: {file}")
            path = os.path.join(anomaly_dir, file)
            score = get_video_max_mse(path, ae_detector)
            y_true.append(1)
            y_scores.append(score)
            
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_scores)
        print("\n--- ConvAE Evaluation Complete ---")
        print(f"ConvAE AUROC: {auroc:.4f}")
    else:
        print("Need both normal and anomalous videos to compute AUROC.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO and ConvAE models pipelines.")
    
    # YOLO arguments
    parser.add_argument("--eval-yolo", action="store_true", help="Run YOLO mAP evaluation")
    parser.add_argument("--yolo-weights", default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--data-yaml", default="dataset.yaml", help="Path to YOLO dataset config (e.g. gun.yaml)")
    
    # Anomaly Detection arguments
    parser.add_argument("--eval-ae", action="store_true", help="Run ConvAE AUROC evaluation")
    parser.add_argument("--ae-weights", default="weights/ae_best.pth", help="Path to ConvAE weights")
    parser.add_argument("--normal-dir", help="Directory of normal videos")
    parser.add_argument("--anomaly-dir", help="Directory of anomalous videos")
    
    args = parser.parse_args()
    
    if not (args.eval_yolo or args.eval_ae):
        parser.print_help()
        print("\nPlease specify --eval-yolo and/or --eval-ae")
        
    if args.eval_yolo:
        evaluate_yolo(args.yolo_weights, args.data_yaml)
        
    if args.eval_ae:
        if not args.normal_dir or not args.anomaly_dir:
            print("Please provide --normal-dir and --anomaly-dir for AE evaluation.")
        else:
            evaluate_anomaly(args.ae_weights, args.normal_dir, args.anomaly_dir)
