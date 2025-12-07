import os
from PIL import Image
from tqdm import tqdm

# ============================================
# 1️⃣ إعداد المسارات
# ============================================
base_dir = "data_unified_64"

# مجلدات الأرقام
digits_dirs = [
    r"C:\Users\hasan\Downloads\Compressed\digits\digits updated",
    r"C:\Users\hasan\Downloads\Compressed\digits\digits_jpeg"
]

# مجلدات الخلفيات
background_dirs = [
    r"C:\Users\hasan\Downloads\Compressed\stanford-background-dataset\images",
    r"C:\Users\hasan\Downloads\Compressed\stanford-background-dataset\labels_colored"
]

# إنشاء مجلد الوجهة والفئات
os.makedirs(base_dir, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)
none_dir = os.path.join(base_dir, "none")
os.makedirs(none_dir, exist_ok=True)


# ============================================
# 2️⃣ دالة حفظ الصورة بعد التحجيم والتحويل
# ============================================
def save_image(src_path, dest_folder):
    try:
        img = Image.open(src_path).convert("L")  # تحويل إلى grayscale
        img = img.resize((64, 64))
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, filename)
        img.save(dest_path)
    except Exception as e:
        print(f"❌ Error processing {src_path}: {e}")


# ============================================
# 3️⃣ معالجة الأرقام (0–9)
# ============================================
print("⚙️ Processing digit datasets...")

for digits_src in digits_dirs:
    for digit in range(10):
        digit_dir = os.path.join(digits_src, str(digit))
        if not os.path.exists(digit_dir):
            print(f"⚠️ Skipping {digit_dir} (not found)")
            continue
        for file in tqdm(os.listdir(digit_dir), desc=f"Digit {digit} from {os.path.basename(digits_src)}"):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                save_image(os.path.join(digit_dir, file), os.path.join(base_dir, str(digit)))


# ============================================
# 4️⃣ معالجة الخلفيات (none)
# ============================================
print("⚙️ Processing background datasets...")

for bg_src in background_dirs:
    if not os.path.exists(bg_src):
        print(f"⚠️ Skipping {bg_src} (not found)")
        continue
    for file in tqdm(os.listdir(bg_src), desc=f"Backgrounds from {os.path.basename(bg_src)}"):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            save_image(os.path.join(bg_src, file), none_dir)


# ============================================
# ✅ النتيجة النهائية
# ============================================
print("🎉 Dataset organized successfully!")
print(f"📂 Final structure saved at: {os.path.abspath(base_dir)}")
