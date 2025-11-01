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
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Import for RAM monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - RAM statistics will be limited")

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5n_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.8
CAMERA_SRC = "http://192.168.0.33:4747/video"  # Android cam
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
DIGIT_MEMORY_TIME = 3  # Increase memory time to 3 seconds
MIN_DIGIT_AREA = 1000  # Minimum area for digit detection
SAVE_INTERVAL = 2.0    # Save ROI every 2 seconds at most
last_save_time = 0
digit = None
number = None

# Statistics variables
start_time = time.time()
total_frames = 0
fps_history = []

########################################
#           YOLO DETECTION FUNCTION
########################################
def run_yolo_detection(frame, yolo_model, confidence_threshold):
    """
    Run YOLO detection on frame and return filtered detections
    """
    results = yolo_model(frame)
    detections = results.pandas().xyxy[0]
    
    filtered_detections = []
    for _, box in detections.iterrows():
        conf = float(box["confidence"])
        if conf >= confidence_threshold:
            x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
            
            # Filter by size (avoid too small detections)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area >= MIN_DIGIT_AREA:
                detection = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf, "area": area
                }
                filtered_detections.append(detection)
    
    return filtered_detections

########################################
#           ROI EXTRACTION FUNCTION
########################################
def extract_and_save_roi(frame, detection, counter, save_dir, current_time):
    """
    Extract ROI from frame based on detection coordinates and save it with rate limiting
    """
    global last_save_time
    
    # Rate limiting - don't save too frequently
    if current_time - last_save_time < SAVE_INTERVAL:
        return None, None, False
    
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    roi = frame[y1:y2, x1:x2]
    
    # Check if ROI is valid
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return None, None, False
    
    # Save ROI
    filename = os.path.join(save_dir, f"Roi_{counter}.png")
    cv2.imwrite(filename, roi)
    last_save_time = current_time
    print(f"📸 Saved ROI {counter} with conf {detection['confidence']:.2f} -> {filename}")
    
    return roi, filename, True

########################################
#           CNN PREDICTION FUNCTION
########################################
def predict_digit_with_cnn(roi, cnn_model, transform, labels):
    """
    Predict digit from ROI using CNN model
    """
    try:
        # Preprocess for CNN model
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1)

            # Handle cases where batch>1
            if probs.dim() > 2:
                probs = probs.mean(dim=0, keepdim=True)

            confidence, predicted = torch.max(probs, dim=1)
            predicted = predicted.view(-1)[0]
            number = labels[predicted.item()]
            conf_num = confidence[0].item()
        
        return number, conf_num
    except Exception as e:
        print(f"❌ CNN prediction error: {e}")
        return "none", 0.0

########################################
#           DIGIT TRACKING FUNCTION
########################################
def is_new_digit(detection, current_time, processed_digits, digit_memory_time):
    """
    Check if detection is a new digit based on spatial-temporal filtering
    """
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Clean up old processed digits
    keys_to_remove = []
    for key, timestamp in processed_digits.items():
        if current_time - timestamp > digit_memory_time:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del processed_digits[key]
    
    # Check if this position was recently processed
    position_key = f"{center_x}_{center_y}"
    for prev_key in processed_digits.keys():
        if position_key in prev_key:
            # Check if it's the same position (within 20 pixels)
            prev_center_x, prev_center_y = map(int, prev_key.split('_')[-2:])
            distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
            if distance < 20:  # Same position within 20 pixels
                return False
    
    return True

########################################
#           DRAW UI FUNCTION
########################################
def draw_ui_panel(frame, detected_digits, processed_digits, current_number, fps):
    """
    Draw user interface panel on frame
    """
    # Draw background panel
    cv2.rectangle(frame, (10, 10), (400, 150), (30, 30, 30), cv2.FILLED)
    
    # Draw detection statistics
    cv2.putText(frame, f"Detections: {len(detected_digits)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    
    # Show tracked digits count
    cv2.putText(frame, f"Tracked: {len(processed_digits)}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    
    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    
    # Show current digit
    digit_text = f"Digit: {current_number}" if current_number else "Digit: None"
    cv2.putText(frame, digit_text,
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

########################################
#           DRAW DETECTIONS FUNCTION
########################################
def draw_detections(frame, detections):
    """
    Draw bounding boxes and confidence on frame
    """
    for detection in detections:
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        conf = detection["confidence"]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

########################################
#           RAM USAGE FUNCTIONS
########################################
def get_ram_usage():
    """Get current RAM usage in GB and percentage"""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            ram_used_gb = memory.used / (1024**3)  # Convert to GB
            ram_total_gb = memory.total / (1024**3)  # Convert to GB
            ram_percent = memory.percent
            return ram_used_gb, ram_total_gb, ram_percent
        except Exception as e:
            print(f"⚠️ RAM monitoring error: {e}")
            return 0, 0, 0
    else:
        return 0, 0, 0

def get_process_ram_usage():
    """Get RAM usage of current process in MB"""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**2)  # Convert to MB
        except Exception as e:
            print(f"⚠️ Process RAM monitoring error: {e}")
            return 0
    else:
        return 0

########################################
#           EXIT STATISTICS FUNCTION
########################################
def print_exit_statistics():
    """Print comprehensive statistics when exiting the program"""
    print("\n" + "="*60)
    print("📊 PROGRAM EXIT STATISTICS")
    print("="*60)
    
    # Calculate runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    minutes = int(total_runtime // 60)
    seconds = int(total_runtime % 60)
    
    print(f"⏱️  Total Runtime: {minutes}m {seconds}s")
    print(f"📈 Total Frames Processed: {total_frames}")
    
    # Calculate average FPS
    if total_runtime > 0:
        avg_fps = total_frames / total_runtime
        print(f"🔄 Average FPS: {avg_fps:.2f}")
    
    # GPU Statistics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        
        print(f"🎮 GPU Device: {gpu_name}")
        print(f"💾 GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"💽 GPU Memory Cached: {gpu_memory_cached:.2f} GB")
    else:
        print("🎮 GPU: Not available (CPU mode)")
    
    # RAM Statistics - System RAM
    if PSUTIL_AVAILABLE:
        ram_used, ram_total, ram_percent = get_ram_usage()
        print(f"🧠 System RAM Usage: {ram_used:.1f}/{ram_total:.1f} GB ({ram_percent:.1f}%)")
        
        # Process-specific RAM usage
        process_ram = get_process_ram_usage()
        print(f"🔍 Process RAM Usage: {process_ram:.1f} MB")
    else:
        print("🧠 RAM Stats: Install 'psutil' for detailed RAM monitoring")
    
    # ROI Statistics
    print(f"📸 ROIs Saved: {counter}")
    print(f"🔢 Unique Digits Tracked: {len(processed_digits)}")
    
    # System performance summary
    if total_runtime > 0:
        frames_per_second = total_frames / total_runtime
        print(f"⚡ Overall Performance: {frames_per_second:.1f} FPS")
    
    # Memory efficiency
    if total_frames > 0:
        memory_per_frame = get_process_ram_usage() / total_frames if PSUTIL_AVAILABLE else 0
        if memory_per_frame > 0:
            print(f"💪 Memory Efficiency: {memory_per_frame:.1f} MB per frame")
    
    print("="*60)
    print("✅ Thank you for using Car Racing Digit Recognition!")
    print("="*60)

########################################
#           DIGIT RECOGNITION LOOP
########################################
def digit_recognition_loop():
    global counter, digit, number, processed_digits, last_save_time, total_frames
    
    # FPS calculation
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Unable to read frame from camera.")
            break

        # Update frame counter for statistics
        total_frames += 1
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Run YOLO detection
        detections = run_yolo_detection(frame, yolo, CONF_THRESHOLD)
        
        detected_digits = []  # store all recognized digits for display
        current_time = time.time()

        # Process each detection
        for detection in detections:
            # Check if this is a new digit
            if is_new_digit(detection, current_time, processed_digits, DIGIT_MEMORY_TIME):
                
                # Extract and save ROI (with rate limiting)
                roi, filename, should_process = extract_and_save_roi(
                    frame, detection, counter, SAVE_DIR, current_time
                )
                
                if not should_process or roi is None:
                    continue
                
                # Predict digit using CNN
                new_number, conf_num = predict_digit_with_cnn(roi, model, transform, labels)
                
                # Only update if confidence is high enough
                if conf_num > 0.7 and new_number != "none":
                    print(f"🔢 Detected: {new_number} (Conf: {conf_num:.2f})")
                    
                    detected_digits.append((new_number, conf_num))
                    
                    # Update global number only if different
                    if new_number != number:
                        number = new_number
                    
                    # Mark this digit as processed
                    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    processed_digits[f"{new_number}_{center_x}_{center_y}"] = current_time
                    
                    counter += 1

        # Draw detections on frame
        draw_detections(frame, detections)
        
        # Draw UI panel
        draw_ui_panel(frame, detected_digits, processed_digits, number, current_fps)

        # Display camera feed
        cv2.imshow("Digit Detection", frame)

        key = cv2.waitKey(1)  # Reduced from 5 to 1 for better responsiveness
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)  # Pause on 's'
        elif key == ord('p'):
            cv2.imwrite("preview.png", frame)
            print("🖼️ Saved snapshot preview.png")
        elif key == ord('c'):
            # Clear processed digits memory
            processed_digits.clear()
            print("🔄 Cleared digit tracking memory")
        elif key == ord('r'):
            # Reset counter
            counter = 0
            print("🔄 Reset ROI counter")

    # Cleanup and print statistics
    cap.release()
    cv2.destroyAllWindows()
    
    # Print exit statistics
    print_exit_statistics()

# Start the main loop

try:
    digit_recognition_loop()
except KeyboardInterrupt:
    print("\n⏹️ Program interrupted by user")
    cap.release()
    cv2.destroyAllWindows()
    print_exit_statistics()
except Exception as e:
    print(f"\n❌ Error: {e}")
    cap.release()
    cv2.destroyAllWindows()
    print_exit_statistics()