##### Car Plate Recognition #####
# Author: Hasan Game
# Description:
# Detects car license plates using YOLOv5 and recognizes each digit
# using a trained CNN model. Works in real time with camera feed.

import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.model import RecognizeNumbersModel
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

########################################
# CONFIGURATION
########################################
YOLO_PLATE_MODEL_PATH = "weights/yolov5n_plates_trained_v1.pt"  # YOLO plate detector
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"            # CNN digits recognizer
CONF_THRESHOLD = 0.7
CAMERA_SRC = "http://192.168.0.33:4747/video"  # Android cam stream
# CAMERA_SRC = 0  # for laptop webcam

SAVE_DIR = "captures_plates"
os.makedirs(SAVE_DIR, exist_ok=True)

sys.path.append("yolov5")

########################################
# MODEL LOADING
########################################
print("🚀 Loading models...")
yolo_plate = torch.hub.load("yolov5", "custom", path=YOLO_PLATE_MODEL_PATH, source="local")

cnn_model = RecognizeNumbersModel()
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True))
cnn_model.eval()

########################################
# IMAGE TRANSFORM (for CNN)
########################################
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

########################################
# CAMERA INITIALIZATION
########################################
cap = cv2.VideoCapture(CAMERA_SRC)
if not cap.isOpened():
    print("❌ Failed to open camera stream.")
    exit()
print("✅ Camera initialized successfully.")

counter = 0
labels = [str(i) for i in range(10)] + ["none"]

# Track recognized plates to avoid duplicate processing
processed_plates = {}
PLATE_MEMORY_TIME = 5.0  # Remember plates for 5 seconds

########################################
# DIGIT SEGMENTATION FUNCTIONS
########################################
def preprocess_plate(plate_roi):
    """Preprocess plate image for digit segmentation"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean the image
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return gray, thresh

def find_digit_contours(thresh, min_height=20, max_height=80, min_aspect=0.2, max_aspect=1.2):
    """Find and filter digit contours"""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        # Filter based on size, aspect ratio and area
        if (min_height <= h <= max_height and 
            min_aspect <= aspect_ratio <= max_aspect and
            area > 50):  # Minimum area to avoid noise
            digit_regions.append((x, y, w, h))
    
    return digit_regions

def recognize_digit(digit_roi, cnn_model, transform, labels):
    """Recognize single digit using CNN model"""
    try:
        digit_img = Image.fromarray(digit_roi)
        img_tensor = transform(digit_img).unsqueeze(0)
        
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1)
            conf_digit, predicted = torch.max(probs, dim=1)
            number = labels[predicted.item()]
            confidence = conf_digit.item()
            
        return number, confidence
    except Exception as e:
        return "none", 0.0

########################################
# MAIN LOOP
########################################
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera.")
        break

    # Clean up old processed plates
    current_time = time.time()
    processed_plates = {plate: timestamp for plate, timestamp in processed_plates.items() 
                       if current_time - timestamp < PLATE_MEMORY_TIME}

    # Run YOLO plate detection
    results = yolo_plate(frame)
    detections = results.pandas().xyxy[0]

    for _, box in detections.iterrows():
        conf = float(box["confidence"])
        if conf < CONF_THRESHOLD:
            continue

        # Extract plate coordinates
        x1, y1, x2, y2 = map(int, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        plate_roi = frame[y1:y2, x1:x2]
        
        # Skip if plate is too small
        if plate_roi.shape[0] < 30 or plate_roi.shape[1] < 100:
            continue

        # Preprocess plate for digit segmentation
        gray_plate, thresh_plate = preprocess_plate(plate_roi)
        
        # Find digit contours
        digit_regions = find_digit_contours(thresh_plate)
        
        # Sort digits from left to right
        digit_regions = sorted(digit_regions, key=lambda r: r[0])
        
        digits_detected = []
        digit_confidences = []

        for i, (x, y, w, h) in enumerate(digit_regions):
            # Extract digit ROI
            digit_crop = gray_plate[y:y+h, x:x+w]
            
            # Recognize digit
            number, confidence = recognize_digit(digit_crop, cnn_model, transform, labels)
            
            # Filter low confidence digits
            if confidence > 0.6 and number != "none":
                digits_detected.append(number)
                digit_confidences.append(confidence)
                
                # Draw rectangle and text for each digit
                cv2.rectangle(plate_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(plate_roi, f"{number}({confidence:.2f})", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Combine recognized digits into plate number
        plate_number = ''.join(digits_detected)
        
        # Only process new plates (not recently seen)
        if plate_number and plate_number not in processed_plates:
            processed_plates[plate_number] = current_time
            
            avg_confidence = np.mean(digit_confidences) if digit_confidences else 0.0
            print(f"🔢 Recognized Plate: {plate_number} (Conf: {avg_confidence:.2f})")
            
            # Save plate image for debugging
            cv2.imwrite(os.path.join(SAVE_DIR, f"plate_{counter}.png"), plate_roi)
            counter += 1

        # Draw YOLO bounding box and plate number
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 3)
        cv2.putText(frame, f"Plate: {plate_number}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    ########################################
    # DISPLAY FRAME WITH ENHANCED UI
    ########################################
    # Display statistics
    cv2.putText(frame, f"Plates detected: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracked plates: {len(processed_plates)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Car Plate Recognition", frame)
    
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite("preview_plate.png", frame)
        print("🖼️ Saved snapshot preview_plate.png")
    elif key == ord('c'):
        # Clear plate memory
        processed_plates.clear()
        print("🔄 Cleared plate tracking memory")

########################################
# CLEANUP
########################################
cap.release()
cv2.destroyAllWindows()
print("✅ Process finished successfully.")