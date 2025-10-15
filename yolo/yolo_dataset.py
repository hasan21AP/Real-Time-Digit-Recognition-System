import os
import random
import cv2

SRC_DIR = "data_unified"
DEST_DIR = "data_yolo"
VAL_SPLIT = 0.2
TARGET_SIZE = (128, 128)

# إنشاء المجلدات المطلوبة
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

# الفئات 0–9 فقط (استبعاد none نهائيًا)
classes = [c for c in sorted(os.listdir(SRC_DIR)) if c.isdigit()]
class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

print("📘 Class IDs:", class_to_id)

def resize_and_save(src, dst):
    img = cv2.imread(src)
    if img is None:
        return
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, resized)

# معالجة كل فئة رقمية
for cls_name in classes:
    cls_path = os.path.join(SRC_DIR, cls_name)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * (1 - VAL_SPLIT))
    train_imgs, val_imgs = imgs[:split_idx], imgs[split_idx:]

    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img_name in split_imgs:
            src_img = os.path.join(cls_path, img_name)
            dst_img = os.path.join(DEST_DIR, f"images/{split}/{img_name}")
            resize_and_save(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            dst_label = os.path.join(DEST_DIR, f"labels/{split}/{label_name}")
            class_id = class_to_id[cls_name]

            with open(dst_label, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("✅ YOLO dataset done", DEST_DIR)
