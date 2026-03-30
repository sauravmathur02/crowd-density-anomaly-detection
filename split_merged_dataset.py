import os
import random
import shutil
from pathlib import Path

def split_dataset(dataset_dir, val_ratio=0.2):
    root = Path(dataset_dir)

    img_train_dir = root / "train" / "images"
    lbl_train_dir = root / "train" / "labels"

    img_val_dir = root / "val" / "images"
    lbl_val_dir = root / "val" / "labels"

    img_val_dir.mkdir(parents=True, exist_ok=True)
    lbl_val_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(img_train_dir.glob("*.*"))

    num_total = len(all_images)
    num_val = int(num_total * val_ratio)

    print(f"Total train images: {num_total}")
    print(f"Moving {num_val} images to validation...")

    val_images = random.sample(all_images, num_val)

    for img_path in val_images:
        lbl_path = lbl_train_dir / (img_path.stem + ".txt")

        target_img = img_val_dir / img_path.name
        target_lbl = lbl_val_dir / (img_path.stem + ".txt")

        shutil.move(str(img_path), str(target_img))

        if lbl_path.exists():
            shutil.move(str(lbl_path), str(target_lbl))

    print("✅ Split completed successfully")


if __name__ == "__main__":
    random.seed(42)
    split_dataset(r"C:\Repo\Crowd and Anomaly Detection\merged_dataset", val_ratio=0.2)