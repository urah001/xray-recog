import os
import shutil
import random
from collections import defaultdict

SOURCE_DIR = "all"
DEST_DIR = "dataset"
TRAIN_DIR = os.path.join(DEST_DIR, "train")
VAL_DIR = os.path.join(DEST_DIR, "val")

# Keywords for classifying
category_keywords = {
    "Cardiomegaly": ["cardiomegaly"],
    "Pneumonia": ["pneumonia"],
    "PulmonaryEdema": ["pulmonaryedema", "edema"],
    "Pneumothorax": ["pneumothorax"],
    "Fracture": ["fracture", "hip", "femur", "humerus", "pelvis"],
    "Normal": ["normal"],
    "Other": []  # default catch-all
}

def get_label(filename):
    filename_lower = filename.lower()
    for label, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return label
    return "Other"

def organize_and_split(train_ratio=0.8):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    grouped_files = defaultdict(list)

    # Group by label
    for filename in os.listdir(SOURCE_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        label = get_label(filename)
        grouped_files[label].append(filename)

    for label, files in grouped_files.items():
        print(f"Label: {label} — {len(files)} images")
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)

        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for subset, subset_files in [("train", train_files), ("val", val_files)]:
            subset_dir = os.path.join(DEST_DIR, subset, label)
            os.makedirs(subset_dir, exist_ok=True)
            for fname in subset_files:
                src = os.path.join(SOURCE_DIR, fname)
                dst = os.path.join(subset_dir, fname)
                shutil.copy(src, dst)

    print("✅ Dataset organized and split into train/val folders.")

if __name__ == "__main__":
    organize_and_split()
