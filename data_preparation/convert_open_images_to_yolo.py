import os
import csv
import shutil
from pathlib import Path

def main():
    # 1. Define paths based on your Open Images structure
    base_dir = Path("datasets/open_images")
    source_images_dir = base_dir / "multidata" / "train" / "Person"
    
    # oidv6 places the CSV files in these folders by default
    bbox_csv_path = base_dir / "boxes" / "oidv6-train-annotations-bbox.csv"
    classes_csv_path = base_dir / "metadata" / "class-descriptions-boxable.csv"
    
    # Fallback in case you moved the CSVs manually into the image folder
    if not bbox_csv_path.exists():
        bbox_csv_path = source_images_dir / "oidv6-train-annotations-bbox.csv"
    if not classes_csv_path.exists():
        classes_csv_path = source_images_dir / "class-descriptions-boxable.csv"

    # Setup unified YOLO output directory architecture
    out_dir = base_dir / "yolo"
    out_img_dir = out_dir / "images"
    out_label_dir = out_dir / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Source Image Directory: {source_images_dir}")
    print(f"📁 Output YOLO Directory: {out_dir}")
    print("-" * 50)
    
    # 2. Extract the Open Images UUID for the "Person" class
    person_label_name = None
    if classes_csv_path.exists():
        with open(classes_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[1] == "Person":
                    person_label_name = row[0]
                    break
    
    # Global default fallback for Person if metadata is missing
    if not person_label_name:
        person_label_name = "/m/01g317"
        print(f"⚠️ Metadata CSV missing. Defaulting Person UUID to {person_label_name}")
    else:
        print(f"✔️ Extracted Person UUID: {person_label_name}")

    if not source_images_dir.exists():
        print(f"❌ Error: Image directory {source_images_dir} does not exist.")
        return

    # 3. Catalog all successfully downloaded images
    available_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in source_images_dir.glob(ext):
            # The downloaded files usually start with 'person_' followed by the ImageID
            # E.g. 'person_038d718a70a24491.jpg' -> extract the raw ID string
            stem = img_path.stem
            image_id = stem.replace("person_", "")
            available_images[image_id] = img_path
            
    print(f"📸 Located {len(available_images)} valid images on disk.")
    if len(available_images) == 0:
        print("Waiting for images to populate... Exiting.")
        return
        
    # 4. Parse the massive CSV and convert Bounding Boxes to YOLO format
    print(f"🔍 Parsing Normalized Annotations from {bbox_csv_path.name}...")
    
    # Dictionary mapping ImageID -> list of YOLO formatted label strings
    yolo_annotations = {img_id: [] for img_id in available_images.keys()}
    
    if not bbox_csv_path.exists():
        print(f"❌ Error: Bounding Box Master CSV {bbox_csv_path} not found.")
        return
        
    with open(bbox_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['LabelName'] != person_label_name:
                continue
                
            img_id = row['ImageID']
            if img_id not in available_images:
                continue  # Protect against memory bloat: skip annotations for images we didn't download
                
            try:
                # Open Images XMin/YMin are ALREADY Normalized between 0.0 and 1.0!
                xmin = float(row['XMin'])
                xmax = float(row['XMax'])
                ymin = float(row['YMin'])
                ymax = float(row['YMax'])
                
                # Mathematical conversion to YOLO format (Center X, Center Y, Width, Height)
                w = xmax - xmin
                h = ymax - ymin
                xc = xmin + (w / 2.0)
                yc = ymin + (h / 2.0)
                
                # Enforce class bounds safely
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                # YOLO Class ID target = 0 for person
                class_id = 0
                yolo_annotations[img_id].append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                
            except ValueError:
                # Skip silently corrupted CSV lines
                continue

    # 5. Export valid pairs to YOLO formatting
    print("⏳ Generating production YOLO format dataset...")
    processed_count = 0
    
    for img_id, img_path in available_images.items():
        annots = yolo_annotations[img_id]
        if not annots:
            continue  # Drop background images safely (if any slipped in)
            
        # Migrate Image
        dest_img = out_img_dir / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
            
        # Write matching text Label file
        dest_label = out_label_dir / f"{img_path.stem}.txt"
        with open(dest_label, 'w') as f:
            f.write("\n".join(annots) + "\n")
            
        processed_count += 1
        
    print("-" * 50)
    print(f"✅ Conversion complete! Processed {processed_count} fully validated image-label pairs.")
    print(f"📂 Your YOLO format data is ready at: {out_dir}")

if __name__ == "__main__":
    main()
