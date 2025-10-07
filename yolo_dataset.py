import os
import shutil
import random
from PIL import Image

# مسار بيانات printed digits
source_dir = r"C:\Users\hasan\.cache\kagglehub\datasets\kshitijdhama\printed-digits-dataset\versions\57\assets"

# المجلد الجديد لشكل YOLO
target_images_train = "data_yolo/images/train"
target_images_val   = "data_yolo/images/val"
target_labels_train = "data_yolo/labels/train"
target_labels_val   = "data_yolo/labels/val"

# إنشاء المجلدات لو مش موجودة
for d in [target_images_train, target_images_val, target_labels_train, target_labels_val]:
    os.makedirs(d, exist_ok=True)

# تقسيم Train/Val بنسبة 80/20
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    class_id = int(class_name)  # المجلد يمثل الرقم
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    random.shuffle(images)

    split_index = int(0.8 * len(images))
    train_imgs = images[:split_index]
    val_imgs   = images[split_index:]

    for phase, img_list in [('train', train_imgs), ('val', val_imgs)]:
        for img_name in img_list:
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            w, h = img.size

            # نسخ الصورة
            shutil.copy(img_path, f"data_yolo/images/{phase}/{img_name}")

            # إنشاء label بنفس الاسم
            label_path = f"data_yolo/labels/{phase}/{img_name.rsplit('.',1)[0]}.txt"

            # الصندوق يغطي كامل الصورة
            x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0

            # كتابة label بصيغة YOLO: class_id x_center y_center width height
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Conversion completed successfully! Check the folder 'data_yolo/'")
