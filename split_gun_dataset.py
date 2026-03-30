import os
import shutil
import random

SOURCE = r"data/Gun/Gunmen Dataset/All"
DEST = r"gun_dataset"

os.makedirs(f"{DEST}/images/train", exist_ok=True)
os.makedirs(f"{DEST}/images/val", exist_ok=True)
os.makedirs(f"{DEST}/labels/train", exist_ok=True)
os.makedirs(f"{DEST}/labels/val", exist_ok=True)

images = [f for f in os.listdir(SOURCE) if f.endswith(".jpg")]
random.shuffle(images)

split_index = int(0.8 * len(images))
train_files = images[:split_index]
val_files = images[split_index:]

def move_files(file_list, split):
    for img in file_list:
        base = img.replace(".jpg", "")
        label = base + ".txt"

        shutil.copy(os.path.join(SOURCE, img),
                    os.path.join(DEST, f"images/{split}", img))

        shutil.copy(os.path.join(SOURCE, label),
                    os.path.join(DEST, f"labels/{split}", label))

move_files(train_files, "train")
move_files(val_files, "val")

print("Train images:", len(train_files))
print("Val images:", len(val_files))
print("Split complete.")