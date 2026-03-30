"""
Print dataset structure for copy-pasting into ChatGPT or docs.
Run: python dataset_structure.py
Then copy the printed output and paste wherever you need it.
"""
import os
import json
from pathlib import Path
from collections import defaultdict

# === CONFIG: point to your data root ===
DATA_ROOT = Path(r"c:\Repo\Crowd and Anomaly Detection\data")
MAX_DEPTH = 5
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def tree(path: Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> list[str]:
    lines = []
    if depth > max_depth:
        return lines
    try:
        items = sorted(
            [p for p in path.iterdir() if p.name != "." and not p.name.startswith(".")]
        )
        dirs = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file()]
        for i, d in enumerate(dirs):
            is_last = i == len(dirs) - 1 and len(files) == 0
            lines.append(f'{prefix}{"└──" if is_last else "├──"} {d.name}/')
            lines.extend(
                tree(d, prefix + ("    " if is_last else "│   "), max_depth, depth + 1)
            )
        for i, f in enumerate(files):
            is_last = i == len(files) - 1
            lines.append(f'{prefix}{"└──" if is_last else "├──"} {f.name}')
    except PermissionError:
        lines.append(f"{prefix}[Permission Denied]")
    except Exception as e:
        lines.append(f"{prefix}[Error: {e}]")
    return lines


def count_by_ext(path: Path) -> dict:
    """Count files by extension per top-level folder under path."""
    counts = defaultdict(lambda: defaultdict(int))
    try:
        for p in path.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(path)
            parts = rel.parts
            folder = parts[0] if parts else "."
            ext = p.suffix.lower() or "(no ext)"
            counts[folder][ext] += 1
    except Exception:
        pass
    return dict(counts)


def summarize_key_files(root: Path) -> list[str]:
    lines = []
    # COCO annotations
    coco_ann = root / "COCO" / "coco2017" / "annotations"
    if coco_ann.exists():
        for j in coco_ann.glob("*.json"):
            try:
                with open(j, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys = list(data.keys())
                lines.append(f"  {j.relative_to(root)}: keys = {keys}")
            except Exception as e:
                lines.append(f"  {j.relative_to(root)}: (read error: {e})")
    # classes.txt
    for classes_file in root.rglob("classes.txt"):
        try:
            with open(classes_file, "r", encoding="utf-8") as f:
                classes = [l.strip() for l in f if l.strip()]
            lines.append(f"  {classes_file.relative_to(root)}: {len(classes)} classes = {classes[:15]}{'...' if len(classes) > 15 else ''}")
        except Exception as e:
            lines.append(f"  {classes_file.relative_to(root)}: (read error: {e})")
    # description.txt
    for desc in root.rglob("description.txt"):
        try:
            with open(desc, "r", encoding="utf-8") as f:
                content = f.read().strip()
            lines.append(f"  {desc.relative_to(root)}: {content}")
        except Exception as e:
            lines.append(f"  {desc.relative_to(root)}: (read error: {e})")
    # Sample one YOLO-style .txt label (class_id x_center y_center width height)
    for label_dir in [root / "Gun" / "Gunmen Dataset" / "All"]:
        if not label_dir.exists():
            continue
        txts = list(label_dir.glob("*.txt"))
        for t in txts[:1]:
            if t.name.lower() == "classes.txt":
                continue
            try:
                with open(t, "r", encoding="utf-8") as f:
                    sample = f.read(200).strip()
                lines.append(f"  Label format (e.g. {t.name}): YOLO-style lines = class_id x_center y_center width height (normalized)")
                lines.append(f"    Sample: {sample[:100]}...")
            except Exception as e:
                lines.append(f"  {t}: (read error: {e})")
        break
    return lines


def main():
    root = DATA_ROOT
    out = []
    out.append("=" * 60)
    out.append("DATASET STRUCTURE (copy everything below for ChatGPT)")
    out.append("=" * 60)
    out.append("")

    if not root.exists():
        out.append(f"Data root not found: {root}")
        print("\n".join(out))
        return

    out.append("1. FOLDER TREE")
    out.append("-" * 40)
    out.append("data/")
    out.extend(tree(root, max_depth=MAX_DEPTH))
    out.append("")

    out.append("2. FILE COUNTS BY FOLDER (extension -> count)")
    out.append("-" * 40)
    for folder, exts in sorted(count_by_ext(root).items()):
        out.append(f"  {folder}/: {dict(exts)}")
    out.append("")

    out.append("3. KEY FILES / FORMATS")
    out.append("-" * 40)
    out.extend(summarize_key_files(root))
    out.append("")

    out.append("4. NOTES")
    out.append("-" * 40)
    out.append("  - COCO: annotations are JSON with keys: info, licenses, images, annotations, categories.")
    out.append("  - Gun: .txt labels are YOLO format (class_id x_center y_center width height), one file per image.")
    out.append("  - Knife: see description.txt for train/test image counts.")
    out.append("")
    out.append("=" * 60)

    text = "\n".join(out)
    print(text)
    # Optionally write to file for easy copy
    report_path = root.parent / "dataset_structure_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"\n(Also saved to: {report_path})")


if __name__ == "__main__":
    main()
