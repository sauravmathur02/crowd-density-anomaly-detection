from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CLASS_MAP = {
    0: "knife",
    1: "gun",
    2: "knife",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate gun/knife classifier crops from a YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        default=r"c:\Repo\Crowd and Anomaly Detection\datasets\combined_v2",
        help="YOLO dataset root containing images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--output-root",
        default=r"c:\Repo\Crowd and Anomaly Detection\project\classifier_data",
        help="Output root for classifier crops.",
    )
    parser.add_argument("--padding", type=float, default=0.18, help="Relative crop padding around each weapon box.")
    parser.add_argument("--min-size", type=int, default=24, help="Minimum crop width/height in pixels after clipping.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="Dataset splits to process.",
    )
    return parser.parse_args()


def find_image_path(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def yolo_to_xyxy(
    label_parts: Iterable[str],
    image_width: int,
    image_height: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    _, cx, cy, w, h = map(float, label_parts[:5])

    box_w = w * image_width
    box_h = h * image_height
    center_x = cx * image_width
    center_y = cy * image_height

    pad_x = box_w * padding
    pad_y = box_h * padding

    x1 = int(round(center_x - (box_w / 2.0) - pad_x))
    y1 = int(round(center_y - (box_h / 2.0) - pad_y))
    x2 = int(round(center_x + (box_w / 2.0) + pad_x))
    y2 = int(round(center_y + (box_h / 2.0) + pad_y))
    return x1, y1, x2, y2


def clip_box(xyxy: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def build_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    padding: float,
    min_size: int,
) -> Dict[str, int]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    split_counter: Counter[str] = Counter()

    if not images_dir.exists() or not labels_dir.exists():
        return dict(split_counter)

    for class_name in CLASS_MAP.values():
        (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image_path(images_dir, label_path.stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        try:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            continue

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            class_name = CLASS_MAP.get(class_id)
            if class_name is None:
                continue

            x1, y1, x2, y2 = clip_box(
                yolo_to_xyxy(parts, img_w, img_h, padding),
                img_w,
                img_h,
            )
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_size or crop_h < min_size:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{image_path.stem}_crop_{idx:03d}.jpg"
            out_path = output_root / split / class_name / out_name
            cv2.imwrite(str(out_path), crop)
            split_counter[class_name] += 1

    return dict(split_counter)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}
    total_counter: Counter[str] = Counter()

    for split in args.splits:
        split_counts = build_split(
            dataset_root=dataset_root,
            output_root=output_root,
            split=split,
            padding=float(args.padding),
            min_size=int(args.min_size),
        )
        summary[split] = split_counts
        total_counter.update(split_counts)
        print(f"{split}: {split_counts}")

    metadata = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "padding": float(args.padding),
        "min_size": int(args.min_size),
        "class_map": CLASS_MAP,
        "summary": summary,
        "total": dict(total_counter),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved metadata to: {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
