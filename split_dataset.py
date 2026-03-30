import os
import random
import shutil
from pathlib import Path

def split_dataset(dataset_dir: str, val_ratio: float = 0.20):
    root = Path(dataset_dir)
    img_train_dir = root / "images" / "train"
    lbl_train_dir = root / "labels" / "train"
    
    img_val_dir = root / "images" / "val"
    lbl_val_dir = root / "labels" / "val"
    
    img_val_dir.mkdir(parents=True, exist_ok=True)
    lbl_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all images in train directory
    all_images = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        all_images.extend(img_train_dir.rglob(f"*{ext}"))
        
    if not all_images:
        print(f"No images found in {img_train_dir}")
        return
        
    num_total = len(all_images)
    num_val = int(num_total * val_ratio)
    
    print(f"Total images found: {num_total}")
    print(f"Moving {num_val} images to validation split...")
    
    # Randomly select validation images
    val_images = random.sample(all_images, num_val)
    
    moved_count = 0
    for img_path in val_images:
        # Corresponding label path
        # Note: image might be inside a subfolder or just directly in train
        rel_path = img_path.relative_to(img_train_dir)
        lbl_path = lbl_train_dir / rel_path.with_suffix(".txt")
        
        target_img = img_val_dir / rel_path
        target_lbl = lbl_val_dir / rel_path.with_suffix(".txt")
        
        # Ensure parent directories exist (if any subfolders)
        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_lbl.parent.mkdir(parents=True, exist_ok=True)
        
        # Move image
        shutil.move(str(img_path), str(target_img))
        
        # Move label if it exists
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(target_lbl))
            
        moved_count += 1
        
    print(f"Successfully moved {moved_count} records to val split.")
    print(f"Training set has {num_total - moved_count} records remaining.")

if __name__ == "__main__":
    dataset_path = r"C:\Repo\Crowd and Anomaly Detection\datasets\combined_v2"
    random.seed(42) # For reproducibility
    split_dataset(dataset_path)
