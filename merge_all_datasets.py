import os
import shutil
import argparse
from glob import glob

# Final class mapping → person=0, weapon=1
CLASS_MAP = {
    "person": 0,
    "gun": 1,
    "knife": 1,
    "weapon": 1,
    "heavy-weapon": 1
}

def read_yaml_classes(yaml_path):
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path, "r") as f:
        lines = f.readlines()
    names = []
    for line in lines:
        if "names" in line:
            names = eval(line.split(":", 1)[1].strip())
    return names

def process_dataset(dataset_path, output_root):
    yaml_path = os.path.join(dataset_path, "data.yaml")
    class_names = read_yaml_classes(yaml_path)

    for split in ["train", "val", "valid"]:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")

        if not os.path.exists(img_dir):
            continue

        out_split = "val" if split in ["val", "valid"] else "train"

        out_img = os.path.join(output_root, out_split, "images")
        out_lbl = os.path.join(output_root, out_split, "labels")

        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        for img_path in glob(os.path.join(img_dir, "*.*")):
            name = os.path.basename(img_path)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")

            if not os.path.exists(lbl_path):
                continue

            new_lines = []

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                if w <= 0 or h <= 0:
                    continue

                if class_names and cls_id < len(class_names):
                    cls_name = class_names[cls_id]
                else:
                    cls_name = "weapon"

                cls_name = cls_name.lower()

                if cls_name not in CLASS_MAP:
                    continue

                new_cls = CLASS_MAP[cls_name]
                new_lines.append(f"{new_cls} {x} {y} {w} {h}\n")

            if not new_lines:
                continue

            new_name = f"{os.path.basename(dataset_path)}_{name}"

            shutil.copy(img_path, os.path.join(out_img, new_name))

            with open(os.path.join(out_lbl, new_name.replace(".jpg", ".txt").replace(".png", ".txt")), "w") as f:
                f.writelines(new_lines)

    print(f"Processed dataset: {dataset_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_dirs = [os.path.join(args.input, d) for d in os.listdir(args.input)]

    for d in dataset_dirs:
        if os.path.isdir(d):
            process_dataset(d, args.output)

    yaml_path = os.path.join(args.output, "data.yaml")

    with open(yaml_path, "w") as f:
        f.write(f"""
train: {os.path.abspath(os.path.join(args.output, "train", "images"))}
val: {os.path.abspath(os.path.join(args.output, "val", "images"))}

nc: 2
names: ['person', 'weapon']
""")

    print("✅ All datasets merged successfully!")

if __name__ == "__main__":
    main()