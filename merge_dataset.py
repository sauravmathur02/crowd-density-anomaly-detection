from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_MAP = {
    0: 0,  # person -> person
    1: 1,  # gun -> weapon
    2: 1,  # knife -> weapon
    3: 1,  # heavy-weapon -> weapon
}
SPLITS = ("train", "val")


@dataclass
class MergeStats:
    images_processed: int = 0
    skipped_files: int = 0
    missing_label_files: int = 0
    empty_label_files: int = 0
    invalid_label_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a YOLO weapon dataset into a 2-class person/weapon dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original YOLO dataset root.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the merged output dataset root.",
    )
    return parser.parse_args()


def find_images(images_dir: Path) -> List[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def remap_label_lines(label_path: Path, stats: MergeStats, class_counter: Counter) -> List[str]:
    if not label_path.exists():
        stats.missing_label_files += 1
        return []

    raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not raw_lines:
        stats.empty_label_files += 1
        return []

    output_lines: List[str] = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 5:
            stats.invalid_label_lines += 1
            continue

        try:
            source_class = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            stats.invalid_label_lines += 1
            continue

        if width <= 0 or height <= 0:
            stats.invalid_label_lines += 1
            continue

        target_class = CLASS_MAP.get(source_class)
        if target_class is None:
            stats.invalid_label_lines += 1
            continue

        remapped = f"{target_class} {x_center} {y_center} {width} {height}"
        output_lines.append(remapped)
        class_counter[target_class] += 1

    return output_lines


def ensure_output_layout(output_root: Path) -> None:
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_dataset_split(input_root: Path, output_root: Path, split: str, stats: MergeStats, class_counter: Counter) -> None:
    images_dir = input_root / "images" / split
    labels_dir = input_root / "labels" / split

    if not images_dir.exists():
        print(f"[WARN] Missing images directory for split '{split}': {images_dir}")
        return

    image_paths = find_images(images_dir)
    if not image_paths:
        print(f"[WARN] No images found for split '{split}': {images_dir}")
        return

    for image_path in image_paths:
        relative_image = image_path.relative_to(images_dir)
        relative_label = relative_image.with_suffix(".txt")
        label_path = labels_dir / relative_label

        output_image_path = output_root / "images" / split / relative_image
        output_label_path = output_root / "labels" / split / relative_label
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(image_path, output_image_path)
        except OSError as exc:
            stats.skipped_files += 1
            print(f"[WARN] Failed to copy image '{image_path}': {exc}")
            continue

        remapped_lines = remap_label_lines(label_path, stats, class_counter)
        output_label_path.write_text(
            "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
            encoding="utf-8",
        )
        stats.images_processed += 1


def write_data_yaml(output_root: Path) -> None:
    yaml_text = (
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 2\n"
        "names: ['person', 'weapon']\n"
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def print_summary(output_root: Path, stats: MergeStats, class_counter: Counter) -> None:
    print("\n=== Merge Summary ===")
    print(f"Total images processed : {stats.images_processed}")
    print(f"Total labels person    : {class_counter[0]}")
    print(f"Total labels weapon    : {class_counter[1]}")
    print(f"Skipped files count    : {stats.skipped_files}")
    print(f"Missing label files    : {stats.missing_label_files}")
    print(f"Empty label files      : {stats.empty_label_files}")
    print(f"Invalid label lines    : {stats.invalid_label_lines}")
    print(f"Output dataset         : {output_root}")
    print(f"data.yaml              : {output_root / 'data.yaml'}")


def validate_input_root(input_root: Path) -> None:
    missing_parts = []
    for split in SPLITS:
        if not (input_root / "images" / split).exists():
            missing_parts.append(f"images/{split}")
        if not (input_root / "labels" / split).exists():
            missing_parts.append(f"labels/{split}")

    if missing_parts:
        print("[WARN] Some expected directories are missing:")
        for item in missing_parts:
            print(f"  - {item}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.exists():
        raise SystemExit(f"Input dataset does not exist: {input_root}")

    ensure_output_layout(output_root)
    validate_input_root(input_root)

    stats = MergeStats()
    class_counter: Counter = Counter()

    for split in SPLITS:
        copy_dataset_split(input_root, output_root, split, stats, class_counter)

    write_data_yaml(output_root)
    print_summary(output_root, stats, class_counter)


if __name__ == "__main__":
    main()
