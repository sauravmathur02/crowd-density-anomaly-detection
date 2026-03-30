import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json, images_dir, output_dir):
    with open(coco_json) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    # Only person (COCO class 1)
    PERSON_ID = 1

    labels = {}

    for ann in data["annotations"]:
        if ann["category_id"] != PERSON_ID:
            continue

        img_id = ann["image_id"]
        bbox = ann["bbox"]  # x,y,w,h

        img_info = images[img_id]
        w_img, h_img = img_info["width"], img_info["height"]

        x, y, w, h = bbox

        # Convert to YOLO format
        x_center = (x + w / 2) / w_img
        y_center = (y + h / 2) / h_img
        w /= w_img
        h /= h_img

        line = f"0 {x_center} {y_center} {w} {h}\n"

        labels.setdefault(img_id, []).append(line)

    os.makedirs(output_dir / "images", exist_ok=True)
    os.makedirs(output_dir / "labels", exist_ok=True)

    for img_id, lines in tqdm(labels.items()):
        img_name = images[img_id]["file_name"]

        src_img = images_dir / img_name
        dst_img = output_dir / "images" / img_name
        dst_lbl = output_dir / "labels" / (Path(img_name).stem + ".txt")

        if not src_img.exists():
            continue

        shutil.copy(src_img, dst_img)

        with open(dst_lbl, "w") as f:
            f.writelines(lines)

    print("✅ COCO person extraction done")


if __name__ == "__main__":
    convert_coco_to_yolo(
    coco_json=Path(r"C:\Repo\Crowd and Anomaly Detection\data\COCO\coco2017\annotations\instances_train2017.json"),
    images_dir=Path(r"C:\Repo\Crowd and Anomaly Detection\data\COCO\coco2017\train2017"),
    output_dir=Path(r"C:\Repo\Crowd and Anomaly Detection\data\coco_person\train")
)