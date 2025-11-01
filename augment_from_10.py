import os, glob, cv2
import albumentations as A

# ---------- PATHS AND PARAMETERS ----------
IN_IMG_DIR  = "capt_plates"            # Directory containing your 10 base images
OUT_IMG_DIR = "data/plates/images"       # Output folder for augmented images
OUT_LBL_DIR = "data/plates/labels"       # Output folder for YOLO label files
AUG_PER_IMAGE = 200                    # Number of augmentations per image (e.g., 100–500)

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# ---------- AUGMENTATION PIPELINE ----------
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.4, p=0.9),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
    A.GaussNoise(var_limit=(5.0, 40.0), p=0.35),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.MedianBlur(blur_limit=3, p=0.1),
    A.Affine(scale=(0.9, 1.05),
             rotate=(-15, 15), shear=(-8, 8), p=0.9),
    A.Perspective(scale=(0.02, 0.06), p=0.35),
    A.ImageCompression(quality_lower=40, quality_upper=95, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.25))

# ---------- HELPER FUNCTIONS ----------
def auto_bbox_from_gray(gray):
    """Automatically find a bounding box around the largest contour (the digit)."""
    # Try binary inverse first (for dark digit on light background)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Try normal binary if inversion fails
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

    # Pick the largest contour (assumed to be the digit)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Add padding to avoid cutting edges
    pad = int(0.06 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)

    H, W = gray.shape
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    ww = w / W
    hh = h / H
    return [xc, yc, ww, hh]

def save_yolo_label(path_lbl, boxes, labels):
    """Save YOLO-format label file (class_id x_center y_center width height)."""
    with open(path_lbl, "w", encoding="utf-8") as f:
        for b, c in zip(boxes, labels):
            f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

# ---------- MAIN EXECUTION ----------
imgs = glob.glob(os.path.join(IN_IMG_DIR, "*.*"))
total_created = 0

for img_p in imgs:
    img = cv2.imread(img_p)
    if img is None:
        print("[WARN] Failed to read:", img_p)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bbox = auto_bbox_from_gray(gray)
    if bbox is None:
        print("[WARN] No digit detected in:", img_p)
        continue

    base = os.path.splitext(os.path.basename(img_p))[0]

    # Save the original image and its label
    out_img0 = os.path.join(OUT_IMG_DIR, f"{base}.jpg")
    out_lbl0 = os.path.join(OUT_LBL_DIR, f"{base}.txt")
    cv2.imwrite(out_img0, img)
    save_yolo_label(out_lbl0, [bbox], [0])  # single class: "number" = 0

    # Generate augmented versions
    created_for_this = 0
    tries = 0
    i = 0

    while created_for_this < AUG_PER_IMAGE and tries < AUG_PER_IMAGE * 2:
        tries += 1
        augmented = transform(image=img, bboxes=[bbox], class_labels=[0])
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        if len(aug_bboxes) == 0:
            continue

        name = f"{base}_aug_{i:04d}"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, name + ".jpg"), aug_img)
        save_yolo_label(os.path.join(OUT_LBL_DIR, name + ".txt"), aug_bboxes, aug_labels)
        i += 1
        created_for_this += 1
        total_created += 1

    print(f"-> From {base}: created {created_for_this} augmented images")

print(f"[DONE] Total augmented images created: {total_created}")
