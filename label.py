import cv2

video = "your_video.mp4"
cap = cv2.VideoCapture(video)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 10 == 0:
        cv2.imwrite(f"frames/frame_{count}.jpg", frame)

    count += 1

cap.release()