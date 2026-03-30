import cv2
import os

def extract(video_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        count += 1

    cap.release()