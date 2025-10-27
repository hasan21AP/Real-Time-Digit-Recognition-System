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
# from yolov5 import YOLOv5
import sys
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5s_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.85
CAMERA_SRC = "http://192.168.0.33:4747/video?fps=60"  # Android cam
# CAMERA_SRC = 0  # for laptop webcam

sys.path.append("yolov5")

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

########################################
#           MODEL LOADING
########################################
print("🚀 Loading models...")
yolo = torch.hub.load("yolov5", "custom", path="weights/yolov5s_trained_v1.pt", source="local")

model = RecognizeNumbersModel()
model.load_state_dict(torch.load(CNN_MODEL_PATH, weights_only=True))
model.eval()

########################################
#           IMAGE TRANSFORMS
########################################
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
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
counter = 0
labels = [str(i) for i in range(10)] + ["none"]

# Track previously detected digits to avoid duplicate processing
processed_digits = {}
DIGIT_MEMORY_TIME = 1  # Remember digits for 3 seconds
digit = None
number = None

########################################
#           MAIN LOOP
########################################
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera.")
        break

    # Run YOLO detection
    results = yolo(frame)
    detections = results.pandas().xyxy[0]


    detected_digits = []  # store all recognized digits for display
    current_time = time.time()

    # Clean up old processed digits
    processed_digits = {digit: timestamp for digit, timestamp in processed_digits.items() 
                       if current_time - timestamp < DIGIT_MEMORY_TIME}

    # Draw YOLO detections and process new digits
    for _, box in detections.iterrows():
        conf = float(box["confidence"])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        

        # Calculate center point of the detection for tracking
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check if this is likely a new digit (not recently processed)
        is_new_digit = True
        for prev_digit, prev_time in processed_digits.items():
            # Simple spatial-temporal filtering - if same position recently processed, skip
            if current_time - prev_time < DIGIT_MEMORY_TIME:  # More aggressive filtering for recent detections
                is_new_digit = False
                break
            if is_new_digit == False:
                break
        if is_new_digit:
            roi = frame[y1:y2, x1:x2]
            # Save ROI
            filename = os.path.join(SAVE_DIR, f"Roi_{counter}.png")
            cv2.imwrite(filename, roi)
            print(f"📸 Saved ROI {counter} with conf {conf:.2f} -> {filename}")

            # Preprocess for CNN model
            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img_pil).unsqueeze(0)
            
            # Show the detected digit
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
                print(f"The digit is: {digit} and the number is: {number}")
                print(f"The condition is : {number == digit}")
                if number == digit:
                    is_new_digit = False
                else:
                    digit = number
                conf_num = confidence[0].item()

            detected_digits.append((number, conf_num))
            print(f"🔢 Detected: {number} (Conf: {conf_num:.2f})")

            # Mark this digit as processed
            processed_digits[f"{number}_{center_x}_{center_y}"] = current_time
            
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
    
    # Show tracked digits count
    cv2.putText(frame, f"Tracked: {len(processed_digits)}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    cv2.putText(frame, f"Digit is: {number}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    if detected_digits:
        last_digit, conf_val = detected_digits[-1]
        cv2.putText(frame, f"Last digit: {last_digit} ({conf_val:.2f})",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
    elif key == ord('c'):
        # Clear processed digits memory
        processed_digits.clear()
        print("🔄 Cleared digit tracking memory")

########################################
#           CLEANUP
########################################
cap.release()
cv2.destroyAllWindows()
print("✅ Process finished successfully.")