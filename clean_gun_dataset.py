import os

# CHANGE THIS PATH
LABELS_DIR = r"data/Gun/Gunmen Dataset/All"   # adjust if needed

GUN_CLASS_ID = 16  # your gun class id from classes.txt

def clean_labels():
    removed_images = 0
    total_files = 0

    for file in os.listdir(LABELS_DIR):
        if file.endswith(".txt") and file != "classes.txt":
            total_files += 1
            path = os.path.join(LABELS_DIR, file)

            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0 and int(parts[0]) == GUN_CLASS_ID:
                    # change class id to 0
                    parts[0] = "0"
                    new_lines.append(" ".join(parts) + "\n")

            if len(new_lines) == 0:
                # no gun in image → delete label + image
                os.remove(path)
                img_path = path.replace(".txt", ".jpg")
                if os.path.exists(img_path):
                    os.remove(img_path)
                removed_images += 1
            else:
                with open(path, "w") as f:
                    f.writelines(new_lines)

    print(f"Total label files checked: {total_files}")
    print(f"Images removed (no gun): {removed_images}")
    print("Cleaning complete.")

if __name__ == "__main__":
    clean_labels()