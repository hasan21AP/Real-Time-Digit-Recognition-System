##### Car Racing Digit Recognition (Optimized Version for CPU) #####
# Author: Hasan Game (optimized by ChatGPT)
# Description:
# Lightweight version of the YOLO + CNN digit recognition system
# optimized for CPU inference with reduced latency and power consumption.


import cv2
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.model import RecognizeNumbersModel
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5n_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.85
CAMERA_SRC = 0  # laptop webcam by default (use phone stream if needed)
# CAMERA_SRC = "http://192.168.0.33:4747/video"

sys.path.append("yolov5")

torch.set_num_threads(1)
cv2.setNumThreads(1)

########################################
#           MODEL LOADING
########################################
print("🚀 Loading models (optimized for CPU)...")

# Load lightweight YOLOv5n model
yolo = torch.hub.load("yolov5", "custom", path=YOLO_MODEL_PATH, source="local", device="cpu")

# Disable FP16 for CPU (it slows down)
if torch.cuda.is_available():
    yolo.half()

# Load CNN classifier
model = RecognizeNumbersModel()
model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

print("✅ Models loaded successfully.")

########################################
#           IMAGE TRANSFORMS
########################################
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),  # smaller input improves speed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

########################################
#           CAMERA INITIALIZATION
########################################
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Failed to open camera stream.")
    exit()

print("✅ Camera initialized successfully.")

counter = 0
labels = [str(i) for i in range(10)] + ["none"]
processed_digits = {}
DIGIT_MEMORY_TIME = 1.0  # seconds
digit = None
number = None
frame_skip = 2  # process every 2nd frame to reduce load

########################################
#           MAIN LOOP
########################################
print("📹 Starting detection loop...")
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to read frame from camera.")
        break

    # Skip frames to improve real-time performance
    if counter % frame_skip != 0:
        counter += 1
        continue

    # YOLO detection with lower input size
    results = yolo(frame, size=224)
    detections = results.pandas().xyxy[0]

    detected_digits = []
    current_time = time.time()

    # Remove outdated detections
    processed_digits = {
        d: t for d, t in processed_digits.items()
        if current_time - t < DIGIT_MEMORY_TIME
    }

    for _, box in detections.iterrows():
        conf = float(box["confidence"])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Skip already processed digits (same position)
        key = f"{center_x}_{center_y}"
        if key in processed_digits:
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            number = labels[predicted.item()]
            conf_num = confidence.item()

        detected_digits.append((number, conf_num))
        processed_digits[key] = current_time

        # Draw detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{number} ({conf_num:.2f})", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # FPS calculation
    fps = 1 / (time.time() - last_time)
    last_time = time.time()

    # Display info panel
    cv2.rectangle(frame, (10, 10), (280, 150), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(detected_digits)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(frame, f"Tracked: {len(processed_digits)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    cv2.putText(frame, f"Digit: {number}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 252, 3), 2)

    cv2.imshow("Digit Detection (Optimized)", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        processed_digits.clear()
        print("🔄 Cleared memory")

    counter += 1

########################################
#           CLEANUP
########################################
cap.release()
cv2.destroyAllWindows()
print("✅ Process finished successfully (Optimized Mode).")
