import os

img_dir = "datasets/combined/images/train"
label_dir = "datasets/combined/labels/train"

removed = 0

for file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, file)

    # check empty file
    if os.path.getsize(label_path) == 0:
        img_file = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_file)

        # delete both
        os.remove(label_path)
        if os.path.exists(img_path):
            os.remove(img_path)

        removed += 1

print(f"Removed {removed} empty samples")