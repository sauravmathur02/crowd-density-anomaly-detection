import random
import shutil
from pathlib import Path

def split_coco(coco_path, val_ratio=0.1):
    train_img = coco_path / "train" / "images"
    train_lbl = coco_path / "train" / "labels"

    val_img = coco_path / "val" / "images"
    val_lbl = coco_path / "val" / "labels"

    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    images = list(train_img.glob("*.*"))
    val_count = int(len(images) * val_ratio)

    val_images = random.sample(images, val_count)

    for img in val_images:
        lbl = train_lbl / (img.stem + ".txt")

        shutil.move(str(img), val_img / img.name)

        if lbl.exists():
            shutil.move(str(lbl), val_lbl / (img.stem + ".txt"))

    print(f"Moved {val_count} images to val")

if __name__ == "__main__":
    random.seed(42)
    split_coco(Path(r"C:\Repo\Crowd and Anomaly Detection\data\New Datasets\coco_person"))