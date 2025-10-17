import os
import random
import cv2
import numpy as np

SRC_DIR = "data_unified"
DEST_DIR = "data_yolo"
VAL_SPLIT = 0.2
TARGET_SIZE = (128, 128)

# إنشاء المجلدات المطلوبة
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

# الفئات 0–9 فقط (استبعاد none)
classes = [c for c in sorted(os.listdir(SRC_DIR)) if c.isdigit()]
class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

print("📘 Class IDs:", class_to_id)

# معالجة كل فئة رقمية
for cls_name in classes:
    cls_path = os.path.join(SRC_DIR, cls_name)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * (1 - VAL_SPLIT))
    train_imgs, val_imgs = imgs[:split_idx], imgs[split_idx:]

    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img_name in split_imgs:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # إنشاء خلفية 512x512 ووضع الصورة في المنتصف
            h, w, _ = img.shape
            canvas = 255 * np.ones((512, 512, 3), dtype=np.uint8)
            x = (512 - w) // 2
            y = (512 - h) // 2
            canvas[y:y+h, x:x+w] = img

            # حفظ الصورة الجديدة
            out_path = os.path.join(DEST_DIR, f"images/{split}", f"{cls_name}_{img_name}")
            cv2.imwrite(out_path, canvas)

            # إنشاء ملف التسمية (YOLO label)
            label_path = os.path.join(DEST_DIR, f"labels/{split}", f"{cls_name}_{img_name.rsplit('.',1)[0]}.txt")
            with open(label_path, "w") as f:
                f.write(f"{class_to_id[cls_name]} 0.5 0.5 1.0 1.0\n")

print("✅ YOLO dataset prepared successfully:", DEST_DIR)
