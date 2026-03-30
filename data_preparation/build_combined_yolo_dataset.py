from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class YoloClassMap:
    person: int = 0
    gun: int = 1
    knife: int = 2


def _iter_image_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield p


def _read_yolo_label_lines(label_path: Path) -> List[str]:
    if not label_path.exists():
        return []
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _write_yolo_label_lines(label_path: Path, lines: Sequence[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _convert_coco_bbox_to_yolo(
    bbox_xywh: Sequence[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    # Clamp for safety.
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return xc, yc, wn, hn


def convert_coco_person_to_yolo_labels(
    coco_instances_json: Path,
    images_dir: Path,
    out_labels_dir: Path,
    class_id_for_person: int,
    max_images: Optional[int] = None,
) -> int:
    """
    Creates YOLO `.txt` label files for COCO images containing `category_id -> person`.

    Writes labels as: `class_id xc yc w h` with normalized floats.
    """
    data = json.loads(coco_instances_json.read_text(encoding="utf-8"))
    categories = data.get("categories", [])

    person_cat_ids = set()
    for c in categories:
        name = str(c.get("name", "")).lower()
        if name == "person":
            person_cat_ids.add(int(c["id"]))

    if not person_cat_ids:
        raise ValueError(f"No COCO category named 'person' found in {coco_instances_json}")

    # Build image metadata maps.
    images = data.get("images", [])
    img_id_to_path: Dict[int, Path] = {}
    img_id_to_size: Dict[int, Tuple[int, int]] = {}
    for im in images:
        img_id = int(im["id"])
        file_name = im["file_name"]
        w = int(im["width"])
        h = int(im["height"])
        img_id_to_path[img_id] = images_dir / file_name
        img_id_to_size[img_id] = (w, h)

    # Gather annotations by image id.
    anns_by_img: Dict[int, List[List[float]]] = {}
    for ann in data.get("annotations", []):
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        if cat_id not in person_cat_ids:
            continue
        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        anns_by_img.setdefault(img_id, []).append([float(x) for x in bbox])

    # Decide which images to process.
    img_items = list(img_id_to_path.items())
    img_items.sort(key=lambda t: t[1].name)
    if max_images is not None:
        img_items = img_items[: int(max_images)]

    written = 0
    # Always create a label file when the image exists, even if it is empty.
    for img_id, img_path in img_items:
        if not img_path.exists():
            continue
        w, h = img_id_to_size[img_id]
        bboxes = anns_by_img.get(img_id, [])
        lines: List[str] = []
        for bbox_xywh in bboxes:
            xc, yc, wn, hn = _convert_coco_bbox_to_yolo(bbox_xywh, w, h)
            lines.append(f"{class_id_for_person} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_path = out_labels_dir / (img_path.stem + ".txt")
        _write_yolo_label_lines(out_path, lines)
        written += 1

    return written


def remap_gun_labels(
    label_lines: Sequence[str],
    src_class_id: int,
    dst_class_id: int,
) -> List[str]:
    out: List[str] = []
    for ln in label_lines:
        parts = ln.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        if cls != src_class_id:
            continue
        parts[0] = str(dst_class_id)
        out.append(" ".join(parts))
    return out


def extract_datasetninja_knife_labels(
    image_path: Path,
    out_label_path: Path,
    class_id_for_knife: int,
) -> None:
    ann_path = image_path.parent.parent / "ann" / (image_path.name + ".json")
    if not ann_path.exists():
        _write_yolo_label_lines(out_label_path, [])
        return
        
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except:
        _write_yolo_label_lines(out_label_path, [])
        return
        
    img_w = data.get("size", {}).get("width", 1)
    img_h = data.get("size", {}).get("height", 1)
    
    lines = []
    for obj in data.get("objects", []):
        if obj.get("classTitle") != "knife":
            continue
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) == 2:
            xmin, ymin = exterior[0]
            xmax, ymax = exterior[1]
            
            w = xmax - xmin
            h = ymax - ymin
            x_center = xmin + w / 2.0
            y_center = ymin + h / 2.0
            
            xc = max(0.0, min(1.0, x_center / img_w))
            yc = max(0.0, min(1.0, y_center / img_h))
            wn = max(0.0, min(1.0, w / img_w))
            hn = max(0.0, min(1.0, h / img_h))
            
            lines.append(f"{class_id_for_knife} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            
    _write_yolo_label_lines(out_label_path, lines)


def build_combined_yolo_dataset(
    out_dir: Path,
    coco_root: Path,
    gun_root: Path,
    knife_images_dir: Path,
    split_seed: int = 1337,
    coco_max_train_images: Optional[int] = None,
    coco_max_val_images: Optional[int] = None,
) -> None:
    """
    Builds a YOLO dataset with classes:
      0 person (from COCO), 1 gun (from gun_dataset), 2 knife (images only; empty labels).

    Output structure:
      out_dir/
        images/train, images/val
        labels/train, labels/val
        data.yaml (written by caller)
    """
    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    class_map = YoloClassMap()

    # 1) COCO person -> YOLO labels.
    coco_instances_train = coco_root / "annotations" / "instances_train2017.json"
    coco_instances_val = coco_root / "annotations" / "instances_val2017.json"
    coco_images_train = coco_root / "train2017"
    coco_images_val = coco_root / "val2017"

    if not coco_instances_train.exists() or not coco_instances_val.exists():
        raise FileNotFoundError("COCO instances JSON not found in expected coco2017/annotations/")

    # Convert labels first (so label files exist).
    convert_coco_person_to_yolo_labels(
        coco_instances_json=coco_instances_train,
        images_dir=coco_images_train,
        out_labels_dir=out_labels_train,
        class_id_for_person=class_map.person,
        max_images=coco_max_train_images,
    )
    convert_coco_person_to_yolo_labels(
        coco_instances_json=coco_instances_val,
        images_dir=coco_images_val,
        out_labels_dir=out_labels_val,
        class_id_for_person=class_map.person,
        max_images=coco_max_val_images,
    )

    # Copy COCO images referenced by label files.
    train_label_files = {p.stem for p in out_labels_train.glob("*.txt")}
    val_label_files = {p.stem for p in out_labels_val.glob("*.txt")}
    for img in _iter_image_files(coco_images_train):
        if img.stem in train_label_files:
            shutil.copy2(img, out_images_train / img.name)
    for img in _iter_image_files(coco_images_val):
        if img.stem in val_label_files:
            shutil.copy2(img, out_images_val / img.name)

    # 2) Gun dataset: YOLO already exists as gun_dataset/images/{train,val} + labels/{train,val}.
    gun_train_images = gun_root / "images" / "train"
    gun_val_images = gun_root / "images" / "val"
    gun_train_labels = gun_root / "labels" / "train"
    gun_val_labels = gun_root / "labels" / "val"
    if not gun_train_images.exists() or not gun_train_labels.exists():
        raise FileNotFoundError(f"Gun dataset not found at {gun_root}. Run split/clean scripts first.")

    # gun labels use class_id=0 in your cleaned set -> remap to 1.
    for img in _iter_image_files(gun_train_images):
        label_in = gun_train_labels / (img.stem + ".txt")
        label_lines = _read_yolo_label_lines(label_in)
        label_out_lines = remap_gun_labels(label_lines, src_class_id=0, dst_class_id=class_map.gun)

        shutil.copy2(img, out_images_train / img.name)
        _write_yolo_label_lines(out_labels_train / (img.stem + ".txt"), label_out_lines)

    for img in _iter_image_files(gun_val_images):
        label_in = gun_val_labels / (img.stem + ".txt")
        label_lines = _read_yolo_label_lines(label_in)
        label_out_lines = remap_gun_labels(label_lines, src_class_id=0, dst_class_id=class_map.gun)

        shutil.copy2(img, out_images_val / img.name)
        _write_yolo_label_lines(out_labels_val / (img.stem + ".txt"), label_out_lines)

    # 3) Knife images: available images only, no YOLO boxes in this repo.
    # We create empty labels for them.
    knife_images = list(_iter_image_files(knife_images_dir))
    if not knife_images:
        # Not fatal; still build person+gun dataset.
        return

    # Split knife images by fixed ratio.
    rng = random.Random(split_seed)
    rng.shuffle(knife_images)
    split_idx = int(0.8 * len(knife_images))
    knife_train = knife_images[:split_idx]
    knife_val = knife_images[split_idx:]

    for img in knife_train:
        shutil.copy2(img, out_images_train / img.name)
        extract_datasetninja_knife_labels(img, out_labels_train / (img.stem + ".txt"), class_map.knife)
    for img in knife_val:
        shutil.copy2(img, out_images_val / img.name)
        extract_datasetninja_knife_labels(img, out_labels_val / (img.stem + ".txt"), class_map.knife)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined YOLO dataset: person+gun+knife.")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory (YOLO format).")
    parser.add_argument("--coco-root", required=True, help="Path to data/COCO/coco2017/")
    parser.add_argument("--gun-root", required=True, help="Path to gun_dataset/ (with images/labels train/val).")
    parser.add_argument("--knife-images-dir", required=True, help="Directory containing knife_*.jpg images.")
    parser.add_argument("--coco-max-train-images", type=int, default=None, help="Optional limit for COCO train images.")
    parser.add_argument("--coco-max-val-images", type=int, default=None, help="Optional limit for COCO val images.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_combined_yolo_dataset(
        out_dir=out_dir,
        coco_root=Path(args.coco_root),
        gun_root=Path(args.gun_root),
        knife_images_dir=Path(args.knife_images_dir),
        coco_max_train_images=args.coco_max_train_images,
        coco_max_val_images=args.coco_max_val_images,
    )

    # Write a YAML config compatible with Ultralytics.
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "  1: gun",
                "  2: knife",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Combined YOLO dataset created at: {out_dir}")
    print(f"Config yaml: {yaml_path}")


if __name__ == "__main__":
    main()

