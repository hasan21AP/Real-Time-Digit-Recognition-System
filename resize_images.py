import cv2
import os

# Path to the original (real) images folder
input_folder = r"E:\Visual_Studio_Work_Place\Cars_Racing_Tracking_System\data_unified"

# Path to save resized images
output_folder = r"E:\Visual_Studio_Work_Place\Cars_Racing_Tracking_System\data_resized"

# Target image size (width, height)
TARGET_SIZE = (256, 256)

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Resize image to the target size
            resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # Keep the same folder structure inside the output directory
            relative_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            # Save the resized image
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, resized)

print("✅ All images have been successfully resized to 256×256!")
