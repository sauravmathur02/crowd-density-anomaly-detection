"""
balance_gun_dataset.py

Oversample gun images/labels in datasets/combined_v2 so that
gun (class 1) count matches knife (class 2) count.
Also adds aggressive augmentation variants (flip, brightness) to diversify.
"""

import os, glob, shutil, random
import cv2
import numpy as np

SRC_IMAGES  = "datasets/combined_v2/images/train"
SRC_LABELS  = "datasets/combined_v2/labels/train"

def count_class(label_dir, cls_id):
    total = 0
    for f in glob.glob(f"{label_dir}/*.txt"):
        for line in open(f).read().strip().splitlines():
            if line and int(line.split()[0]) == cls_id:
                total += 1
    return total

def get_files_with_class(label_dir, cls_id):
    result = []
    for f in glob.glob(f"{label_dir}/*.txt"):
        content = open(f).read().strip()
        for line in content.splitlines():
            if line and int(line.split()[0]) == cls_id:
                result.append(f)
                break
    return result

gun_count   = count_class(SRC_LABELS, 1)
knife_count = count_class(SRC_LABELS, 2)
print(f"Gun (1): {gun_count}  |  Knife (2): {knife_count}")
needed = knife_count - gun_count

if needed <= 0:
    print("Gun and knife are already balanced or gun is more! Nothing to do.")
    exit()

print(f"Need to create {needed} additional gun label instances...")

gun_label_files = get_files_with_class(SRC_LABELS, 1)
print(f"Source gun files available: {len(gun_label_files)}")

created = 0
iteration = 0

while created < needed:
    iteration += 1
    random.shuffle(gun_label_files)
    for lbl_path in gun_label_files:
        if created >= needed:
            break
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        
        # Find matching image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(SRC_IMAGES, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Apply a unique augmentation per iteration
        aug_suffix = f"_gunbal_{iteration}_{created}"
        
        if iteration % 3 == 0:
            # Horizontal flip (flip labels too)
            aug_img = cv2.flip(img, 1)
            labels_raw = open(lbl_path).read().strip().splitlines()
            aug_labels = []
            for line in labels_raw:
                parts = line.strip().split()
                if not parts: continue
                c, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cx = 1.0 - cx  # flip x center
                aug_labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        elif iteration % 3 == 1:
            # Brightness/contrast jitter
            alpha = random.uniform(0.75, 1.25)
            beta  = random.randint(-20, 20)
            aug_img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            aug_labels = open(lbl_path).read().strip().splitlines()
        else:
            # Small rotation
            h_px, w_px = img.shape[:2]
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w_px/2, h_px/2), angle, 1.0)
            aug_img = cv2.warpAffine(img, M, (w_px, h_px))
            aug_labels = open(lbl_path).read().strip().splitlines()  # approx same bbox
            
        # Save augmented image and labels
        img_ext = os.path.splitext(img_path)[1]
        new_img_path = os.path.join(SRC_IMAGES, stem + aug_suffix + img_ext)
        new_lbl_path = os.path.join(SRC_LABELS, stem + aug_suffix + ".txt")
        
        cv2.imwrite(new_img_path, aug_img)
        if isinstance(aug_labels, list):
            open(new_lbl_path, 'w').write('\n'.join(aug_labels))
        else:
            open(new_lbl_path, 'w').write(aug_labels)
        
        created += 1
        if created % 100 == 0:
            print(f"  Created {created}/{needed}")

print(f"\nDone! Created {created} new gun augmentation samples.")
print(f"\nVerify counts:")
print(f"  Gun (1): {count_class(SRC_LABELS, 1)}")
print(f"  Knife (2): {count_class(SRC_LABELS, 2)}")
print(f"\nNow retrain with:")
print('  yolo task=detect mode=train model=yolov8m.pt data=datasets/combined_v2/dataset.yaml epochs=100 imgsz=960 batch=8 device=0 cls=2.0')
