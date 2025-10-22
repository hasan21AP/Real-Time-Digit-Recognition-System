##### Car Racing Digit Recognition #####
# Author: Hasan Game (based on EJ Technology Consultants structure)
# Description:
# This script uses a custom YOLO model to detect printed numbers in live camera frames.
# Then, each detected ROI (digit area) is processed and recognized using a trained CNN model.
# The result is displayed in real-time with confidence values.

import os
import cv2
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.model import RecognizeNumbersModel
from ultralytics import YOLO

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/best.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.5
CAPTURE_INTERVAL = 2.0
CAMERA_SRC = "http://192.168.0.33:4747/video?fps=60"  # Android cam
# CAMERA_SRC = 0  # for laptop webcam

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

########################################
#           MODEL LOADING
########################################
print("🚀 Loading models...")
yolo = YOLO(YOLO_MODEL_PATH)

model = RecognizeNumbersModel()
model.load_state_dict(torch.load(CNN_MODEL_PATH, weights_only=True))
model.eval()

########################################
#           IMAGE TRANSFORMS
########################################
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

########################################
#           CAMERA INITIALIZATION
########################################
cap = cv2.VideoCapture(CAMERA_SRC)
if not cap.isOpened():
    print("❌ Failed to open camera stream.")
    exit()

print("✅ Camera initialized successfully.")
last_capture_time = 0
counter = 0
labels = [str(i) for i in range(10)] + ["none"]

########################################
#           MAIN LOOP
########################################
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera.")
        break

    # Run YOLO detection
    results = yolo(frame, verbose=False)
    detections = results[0].boxes

    detected_digits = []  # store all recognized digits for display

    # Draw YOLO detections
    for box in detections:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        # Process ROI every few seconds
        if time.time() - last_capture_time > CAPTURE_INTERVAL:
            last_capture_time = time.time()

            # Save ROI
            filename = os.path.join(SAVE_DIR, f"Roi_{counter}.png")
            cv2.imwrite(filename, roi)
            print(f"📸 Saved ROI {counter} with conf {conf:.2f} -> {filename}")

            # Preprocess for CNN model

            img_pil = Image.fromarray(roi)
            img_tensor = transform(img_pil).unsqueeze(0)
            filename = os.path.join(SAVE_DIR, f"Tensor_{counter}.png")
            cv2.imshow("Digit", roi)
            

            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)

                # Handle cases where batch>1
                if probs.dim() > 2:
                    probs = probs.mean(dim=0, keepdim=True)

                confidence, predicted = torch.max(probs, dim=1)
                predicted = predicted.view(-1)[0]
                number = labels[predicted.item()]
                conf_num = confidence[0].item()

            detected_digits.append((number, conf_num))
            print(f"🔢 Detected: {number} (Conf: {conf_num:.2f})")

            counter += 1

        # Draw bounding box on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ########################################
    #           DRAW UI PANEL
    ########################################
    cv2.rectangle(frame, (10, 10), (360, 110), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, f"Detections: {len(detected_digits)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    if detected_digits:
        last_digit, conf_val = detected_digits[-1]
        cv2.putText(frame, f"Last digit: {last_digit} ({conf_val:.2f})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display camera feed
    cv2.imshow("Digit Detection", frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey()
    elif key == ord('p'):
        cv2.imwrite("preview.png", frame)
        print("🖼️ Saved snapshot preview.png")

########################################
#           CLEANUP
########################################
cap.release()
cv2.destroyAllWindows()
print("✅ Process finished successfully.")
