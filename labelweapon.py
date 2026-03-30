import cv2
import os

video_path = "Kid_with_Knife_vs._Man_with_Gun_instructional_training_airsoft_not_real_720P.mp4"  # <-- change this
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Video not opened")
    exit()

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 10 == 0:
        file_path = os.path.join(output_dir, f"frame_{saved}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")
        saved += 1

    count += 1

cap.release()

print(f"✅ Done! Total frames saved: {saved}")