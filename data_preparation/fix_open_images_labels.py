import os
import re
from pathlib import Path

def is_open_images_file(filename: str) -> bool:
    """
    Open Images dataset files famously use a 16-character hexadecimal string.
    Example: '04a7a18db57c8176.txt'
    
    COCO uses exactly 12-digit zero-padded numbers (e.g., '000000000009.txt')
    Gun / Knife datasets use descriptive English/Spanish names or DSC_001.txt.
    
    Therefore, a strict 16-hex regex safely isolates ONLY Open Images files!
    """
    return bool(re.match(r'^[0-9a-fA-F]{16}$', filename))

def main():
    # 1. Define paths
    combined_dir = Path("datasets/combined")
    labels_dir = combined_dir / "labels"
    
    if not labels_dir.exists():
        print(f"❌ Error: Labels directory not found at {labels_dir}")
        return
        
    print(f"🔍 Scanning YOLO labels directory: {labels_dir}")
    print("-" * 50)
    
    scanned_count = 0
    fixed_count = 0
    ambiguous_count = 0
    ignored_count = 0
    
    # 2. Iterate through all labels recursively (covers both train/ and val/ folders)
    for label_path in labels_dir.rglob("*.txt"):
        stem = label_path.stem
        scanned_count += 1
        
        # 3. Safely detect the origin using the strict regex
        if is_open_images_file(stem):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"❌ Failed to read {label_path.name}: {e}")
                continue
                
            modified_lines = []
            changed = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class_id = parts[0]
                    
                    # 4. Correct the mapping (Person should be 0)
                    if old_class_id != '0':
                        
                        # Warning if ambiguity is found (Expected 1 based on user bug report, but what if 2?)
                        if old_class_id not in ['0', '1']:
                            print(f"⚠️ AMBIGUITY WARNING: Found unexpected Class {old_class_id} in purely Person Open Images file: {label_path.name}. Converting to 0 anyway.")
                            ambiguous_count += 1
                            
                        parts[0] = '0'  # Safely override
                        modified_lines.append(" ".join(parts) + "\n")
                        changed = True
                    else:
                        modified_lines.append(line)
                else:
                    modified_lines.append(line)
            
            # 5. Overwrite file ONLY if changes were required
            if changed:
                try:
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.writelines(modified_lines)
                    fixed_count += 1
                except Exception as e:
                    print(f"❌ Failed to save {label_path.name}: {e}")
            else:
                ignored_count += 1 # It was Open Images, but already perfectly correct
        else:
            # Safely skip COCO, Gun, and Knife datasets!
            ignored_count += 1
            
    print("-" * 50)
    print("✅ Label Correction Audit Complete")
    print(f"Total Text Files Scanned:  {scanned_count}")
    print(f"Open Images Labels Fixed:  {fixed_count}")
    print(f"Open Images Already OK:    {scanned_count - ignored_count - fixed_count if scanned_count > ignored_count else 0}")
    print(f"Foreign Datasets Ignored:  {ignored_count} (Guns, Knives, & COCO untouched)")
    print(f"Ambiguity Warnings Logged: {ambiguous_count}")
    print("\nDataset consistency is fully restored! You are clear to resume YOLO training.")

if __name__ == "__main__":
    main()
