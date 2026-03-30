from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")
print("Model classes mapping:", model.names)
print("Testing on a dummy image...")
results = model("datasets/combined_v2/images/train/1.jpg") # try finding an image or just a zeros array
