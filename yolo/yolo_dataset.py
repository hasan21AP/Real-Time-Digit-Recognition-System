import os
import random
import shutil
from difflib import get_close_matches

# === Configuration ===
SRC_DIR = "yolo_data_v2"              # Folder that contains images/ and labels/
DEST_DIR = "yolo_data_ready_v2"       # Output folder
VAL_SPLIT = 0.2                    # 20% for validation

# === Create destination folders ===
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

# === Collect all image files ===
images_dir = os.path.join(SRC_DIR, "images")
labels_dir = os.path.join(SRC_DIR, "labels")

all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

split_index = int(len(all_images) * (1 - VAL_SPLIT))
train_imgs = all_images[:split_index]
val_imgs = all_images[split_index:]

def find_matching_label(img_name, all_labels):
    """Finds the most similar label filename based on partial match."""
    name_root = os.path.splitext(img_name)[0]
    matches = get_close_matches(name_root, all_labels, n=1, cutoff=0.3)
    return matches[0] if matches else None

def copy_pair(img_list, split):
    """Copy images and matching labels (even if names differ slightly)."""
    all_labels = os.listdir(labels_dir)
    copied, missing = 0, 0

    for img_name in img_list:
        img_src = os.path.join(images_dir, img_name)
        img_dst = os.path.join(DEST_DIR, f"images/{split}", img_name)
        shutil.copy(img_src, img_dst)

        match_label = find_matching_label(img_name, all_labels)
        if match_label:
            label_src = os.path.join(labels_dir, match_label)
            label_dst = os.path.join(DEST_DIR, f"labels/{split}", os.path.splitext(img_name)[0] + ".txt")
            shutil.copy(label_src, label_dst)
            copied += 1
        else:
            print(f"⚠️ Missing label for {img_name}")
            missing += 1

    print(f"✅ {split.capitalize()} set: {copied} matched, {missing} missing")

# === Copy training and validation data ===
copy_pair(train_imgs, "train")
copy_pair(val_imgs, "val")

print(f"\n✅ YOLO dataset prepared successfully in '{DEST_DIR}'")
print(f"📊 Train: {len(train_imgs)}, Val: {len(val_imgs)}")
