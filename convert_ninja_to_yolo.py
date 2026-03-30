import json
import os
import shutil
from pathlib import Path

def convert_ninja_to_yolo(ninja_root: str):
    root = Path(ninja_root)
    ann_dir = root / "ds" / "ann"
    img_dir = root / "ds" / "img"
    
    if not ann_dir.exists() or not img_dir.exists():
        print(f"Skipping {ninja_root}, missing 'ds/ann' or 'ds/img'")
        return

    # Target directories
    labels_out = root / "labels"
    images_out = root / "images"
    labels_out.mkdir(exist_ok=True)
    images_out.mkdir(exist_ok=True)

    CLASS_MAP = {
        "person": 0,
        "pistol": 1,
        "gun": 1,
        "handgun": 1,
        "rifle": 1,
        "firearm": 1,
        "knife": 2
    }

    count = 0
    for img_path in img_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
            
        json_path = ann_dir / f"{img_path.name}.json"
        if not json_path.exists():
            continue

        with json_path.open() as f:
            data = json.load(f)
            
        img_w = data["size"]["width"]
        img_h = data["size"]["height"]
        
        if img_w == 0 or img_h == 0:
            continue
            
        yolo_lines = []
        for obj in data.get("objects", []):
            if obj["geometryType"] != "rectangle":
                continue
                
            class_title = obj.get("classTitle", "").lower()
            
            # Map class
            yolo_id = None
            for key, val in CLASS_MAP.items():
                if key in class_title:
                    yolo_id = val
                    break
            
            if yolo_id is None:
                continue
                
            exterior = obj["points"]["exterior"]
            if len(exterior) != 2:
                continue
                
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            
            # Ensure proper top-left, bottom-right
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Normalize
            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            box_w = (x_max - x_min) / img_w
            box_h = (y_max - y_min) / img_h
            
            yolo_lines.append(f"{yolo_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            
        if not yolo_lines:
            continue
            
        # Write files
        base_name = img_path.stem
        shutil.copy2(img_path, images_out / img_path.name)
        with open(labels_out / f"{base_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
            
        count += 1

    print(f"Converted {count} images and labels in {ninja_root}")


if __name__ == '__main__':
    base_dir = Path(r"C:\Repo\Crowd and Anomaly Detection\data\New Datasets")
    for d in base_dir.iterdir():
        if d.is_dir() and "DatasetNinja" in d.name:
            print(f"Processing {d.name}...")
            convert_ninja_to_yolo(str(d))
