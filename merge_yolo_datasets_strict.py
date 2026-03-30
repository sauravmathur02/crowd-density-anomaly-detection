#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
merge_yolo_datasets_strict.py

Merge multiple YOLO‑format datasets into one unified dataset with a
strict class mapping:

    0 → person
    1 → gun   (pistol, handgun, rifle, firearm, any weapon‑related)
    2 → knife

All other classes are discarded.

Features
--------
* Handles the common layout `images/` & `labels/` inside each dataset.
* Supports .jpg, .jpeg, .png image extensions.
* Prevents filename collisions by prefixing every file with the
  originating dataset name (e.g. `pistol_image_001.jpg`).
* No heuristic guessing – class mapping is driven **only** by the
  dataset folder name.
* Skips empty label files and images that end up with no valid objects.
* Produces a concise log with totals and per‑class counts.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path
import sys

# ----------------------------------------------------------------------
# 1️⃣  Configuration (change only if you move the repo)
# ----------------------------------------------------------------------
ROOT = Path(r"C:\Repo\Crowd and Anomaly Detection")
DATA_ROOT = ROOT / "data" / "New Datasets"
OUT_ROOT = ROOT / "datasets" / "combined_v2"

IMG_OUT = OUT_ROOT / "images" / "train"
LBL_OUT = OUT_ROOT / "labels" / "train"

# ----------------------------------------------------------------------
# 2️⃣  Deterministic class mapping based on dataset name
# ----------------------------------------------------------------------
# The key must match the **folder name** of each source dataset (case‑insensitive)
DATASET_CLASS_MAP = {
    "pistol":   {"person": 0, "pistol": 1, "handgun": 1, "rifle": 1,
                 "firearm": 1, "weapon": 1},
    "knife":    {"person": 0, "knife": 2, "blade": 2},
    "weapon":   {"person": 0, "pistol": 1, "handhand": 1, "rifle": 1,
                 "firearm": 1, "weapon": 1},
    "sohas":    {"person": 0, "pistol": 1, "handgun": 1, "rifle": 1,
                 "firearm": 1, "weapon": 1, "knife": 2, "blade": 2},
    "default": {"person": 0, "gun": 1, "pistol": 1, "handgun": 1,
                "rifle": 1, "firearm": 1, "weapon": 1,
                "knife": 2, "blade": 2},
}

# ----------------------------------------------------------------------
# 3️⃣  Helper utilities
# ----------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def get_dataset_mapping(dataset_name: str) -> dict:
    """Return the class‑name → target‑id dict for a given dataset folder."""
    lower = dataset_name.lower()
    for key, mapping in DATASET_CLASS_MAP.items():
        if key in lower:
            return mapping
    return DATASET_CLASS_MAP["default"]

def get_class_names(dataset_dir: Path) -> list[str] | None:
    """Attempt to find class names in classes.txt or data.yaml."""
    for classes_file in dataset_dir.rglob("classes.txt"):
        if classes_file.is_file():
            with classes_file.open("r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    for yaml_file in dataset_dir.rglob("*.yaml"):
        if yaml_file.is_file():
            with yaml_file.open("r", encoding="utf-8") as f:
                content = f.read()
                import re
                match = re.search(r"names:\s*\[(.*?)\]", content)
                if match:
                    names_str = match.group(1)
                    return [x.strip().strip("'\"") for x in names_str.split(",")]
    return None

def find_image_file(label_path: Path) -> Path | None:
    """Locate the image file corresponding to a YOLO label file.
    Expected layout: <dataset>/train/labels/*.txt and <dataset>/train/images/.
    """
    stem = label_path.stem
    parts = list(label_path.parts)
    if "labels" in parts:
        # Find the last occurrence of 'labels' and replace with 'images'
        idx = len(parts) - 1 - parts[::-1].index("labels")
        parts[idx] = "images"
        images_dir = Path(*parts[:-1])  # Exclude the filename
        if images_dir.is_dir():
            for ext in IMG_EXTS:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate

    # Fallback: search any matching image under a reasonable dataset root
    # Go up 3 levels from labels (e.g., train -> dataset_root)
    if len(label_path.parents) >= 3:
        dataset_root = label_path.parents[2]
        for ext in IMG_EXTS:
            for candidate in dataset_root.rglob(f"{stem}{ext}"):
                if candidate.is_file():
                    return candidate
    return None

def read_label_file(label_path: Path) -> list[str]:
    with label_path.open("r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def parse_yolo_line(line: str) -> tuple[int, list[str]]:
    parts = line.split()
    return int(parts[0]), parts[1:]

def write_label_file(dst_path: Path, lines: list[str]) -> None:
    with dst_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

def unique_name(prefix: str, original_name: str) -> str:
    return f"{prefix}_{original_name}"

# ----------------------------------------------------------------------
# 4️⃣  Main merging routine
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets into a strict "
                    "person/gun/knife unified dataset."
    )
    parser.add_argument("--src", type=Path, default=DATA_ROOT,
                        help="Root directory containing the individual YOLO datasets.")
    parser.add_argument("--dst", type=Path, default=OUT_ROOT,
                        help="Destination directory for the combined dataset.")
    args = parser.parse_args()

    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    total_copied = 0
    total_skipped = 0
    class_counter = Counter()

    for dataset_dir in sorted(args.src.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        mapping = get_dataset_mapping(dataset_name)
        # Recursively find all YOLO label files (*.txt) within the dataset.
        label_paths = []
        for lbl_dir in dataset_dir.rglob("labels"):
            if lbl_dir.is_dir():
                label_paths.extend(lbl_dir.rglob("*.txt"))
        if not label_paths:
            print(f"[WARN] No label files found in {dataset_dir}, skipping.")
            continue
        for label_path in sorted(label_paths):
            total_seen += 1
            # Locate the corresponding image.
            img_path = find_image_file(label_path)
            if img_path is None:
                total_skipped += 1
                continue

            # Load class names for this dataset once
            dataset_class_names = get_class_names(dataset_dir)

            # Read and remap labels.
            raw_lines = read_label_file(label_path)
            new_lines = []
            for line in raw_lines:
                orig_cls, coords = parse_yolo_line(line)
                class_name = None
                if dataset_class_names and 0 <= orig_cls < len(dataset_class_names):
                    class_name = dataset_class_names[orig_cls].lower()
                
                if class_name is None:
                    class_name = str(orig_cls).lower()
                target_id = mapping.get(class_name)
                if target_id is None:
                    continue
                new_lines.append(f"{target_id} " + " ".join(coords))
                class_counter[target_id] += 1

            if not new_lines:
                total_skipped += 1
                continue

            # Unique filenames to avoid collisions.
            uniq_img = unique_name(dataset_name, img_path.name)
            uniq_lbl = unique_name(dataset_name, label_path.name)
            shutil.copy2(img_path, IMG_OUT / uniq_img)
            write_label_file(LBL_OUT / uniq_lbl, new_lines)
            total_copied += 1


    print("\n=== Merge Summary ===")
    print(f"Source root          : {args.src}")
    print(f"Destination root     : {args.dst}")
    print(f"Datasets processed   : {len(list(args.src.iterdir()))}")
    print(f"Total label files examined : {total_seen}")
    print(f"Images copied        : {total_copied}")
    print(f"Images skipped (missing image / empty after filtering) : {total_skipped}")
    print("\nClass distribution in the combined set:")
    for cid in (0, 1, 2):
        print(f"  class {cid}: {class_counter[cid]} objects")
    print("\nUnified dataset layout:")
    print(f"  Images → {IMG_OUT}")
    print(f"  Labels → {LBL_OUT}")
    print("\nAll done! You can now point your YOLO training script to the folder:")
    print(f"  {OUT_ROOT}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
