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

# Import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - CPU/RAM statistics will be limited")

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5n_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.7
CAMERA_SRC = 0  # laptop webcam by default (use phone stream if needed)
ANDROID_SRC = "http://192.168.0.33:4747/video"
# CAMERA_SRC = "http://192.168.0.33:4747/video"

sys.path.append("yolov5")

torch.set_num_threads(1)
cv2.setNumThreads(1)

########################################
#           GLOBAL VARIABLES
########################################
counter = 0
labels = [str(i) for i in range(10)] + ["none"]
processed_digits = {}
DIGIT_MEMORY_TIME = 1.0  # seconds
current_digit = None
frame_skip = 0.2  # process every 2nd frame to reduce load

# Statistics variables
start_time = time.time()
total_frames_processed = 0
fps_history = []

########################################
#           PERFORMANCE MONITORING FUNCTIONS
########################################
def get_cpu_usage():
    """Get current CPU usage percentage"""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            print(f"⚠️ CPU monitoring error: {e}")
            return 0
    return 0

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
    return 0

def print_exit_statistics():
    """Print comprehensive statistics when exiting the program"""
    print("\n" + "="*60)
    print("📊 PROGRAM EXIT STATISTICS - CPU OPTIMIZED")
    print("="*60)
    
    # Calculate runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    minutes = int(total_runtime // 60)
    seconds = int(total_runtime % 60)
    
    print(f"⏱️  Total Runtime: {minutes}m {seconds}s")
    print(f"📈 Total Frames Processed: {total_frames_processed}")
    
    # Calculate average FPS
    if total_runtime > 0:
        avg_fps = total_frames_processed / total_runtime
        print(f"🔄 Average FPS: {avg_fps:.2f}")
    
    # CPU Statistics
    cpu_usage = get_cpu_usage()
    print(f"🔧 CPU Usage: {cpu_usage:.1f}%")
    
    # RAM Statistics - System RAM
    if PSUTIL_AVAILABLE:
        ram_used, ram_total, ram_percent = get_ram_usage()
        print(f"🧠 System RAM Usage: {ram_used:.1f}/{ram_total:.1f} GB ({ram_percent:.1f}%)")
        
        # Process-specific RAM usage
        process_ram = get_process_ram_usage()
        print(f"🔍 Process RAM Usage: {process_ram:.1f} MB")
    else:
        print("🧠 RAM Stats: Install 'psutil' for detailed RAM monitoring")
    
    # Performance Statistics
    print(f"📸 Digits Detected: {counter}")
    print(f"🔢 Unique Digits Tracked: {len(processed_digits)}")
    
    # Efficiency metrics
    if total_frames_processed > 0:
        frames_per_second = total_frames_processed / total_runtime
        print(f"⚡ Overall Performance: {frames_per_second:.1f} FPS")
        
        if PSUTIL_AVAILABLE:
            memory_per_frame = get_process_ram_usage() / total_frames_processed
            print(f"💪 Memory Efficiency: {memory_per_frame:.1f} MB per frame")
    
    print("="*60)
    print("✅ Thank you for using Car Racing Digit Recognition!")
    print("="*60)

########################################
#           MODEL LOADING FUNCTION
########################################
def load_models():
    """Load YOLO and CNN models optimized for CPU"""
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
    return yolo, model

########################################
#           IMAGE TRANSFORMS FUNCTION
########################################
def get_image_transforms():
    """Define image transformations for CNN"""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),  # smaller input improves speed
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

########################################
#           CAMERA INITIALIZATION FUNCTION
########################################
def initialize_camera():
    """Initialize camera with optimized settings"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Failed to open camera stream.")
        sys.exit(1)
    
    print("✅ Camera initialized successfully.")
    return cap

########################################
#           YOLO DETECTION FUNCTION
########################################
def run_yolo_detection(yolo_model, frame):
    """
    Run YOLO detection on frame and return detections
    """
    results = yolo_model(frame, size=224)
    detections = results.pandas().xyxy[0]
    
    filtered_detections = []
    for _, box in detections.iterrows():
        conf = float(box["confidence"])
        if conf < CONF_THRESHOLD:
            continue
        
        x1, y1, x2, y2 = map(int, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        filtered_detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_x": center_x, "center_y": center_y,
            "confidence": conf
        })
    
    return filtered_detections

########################################
#           ROI EXTRACTION FUNCTION
########################################
def extract_roi(frame, detection):
    """
    Extract Region of Interest from frame based on detection
    """
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
    
    return roi

########################################
#           DIGIT TRACKING FUNCTION
########################################
def is_new_digit(detection, current_time):
    """
    Check if detection is a new digit based on spatial-temporal filtering
    """
    global processed_digits
    
    key = f"{detection['center_x']}_{detection['center_y']}"
    
    # Remove outdated detections
    processed_digits = {
        d: t for d, t in processed_digits.items()
        if current_time - t < DIGIT_MEMORY_TIME
    }
    
    return key not in processed_digits

########################################
#           CNN PREDICTION FUNCTION
########################################
def predict_digit_with_cnn(roi, cnn_model, transform):
    """
    Predict digit from ROI using CNN model
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)
        
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            number = labels[predicted.item()]
            conf_num = confidence.item()
        
        return number, conf_num
    except Exception as e:
        print(f"❌ CNN prediction error: {e}")
        return "none", 0.0

########################################
#           DRAW UI FUNCTION
########################################
def draw_ui_panel(frame, detected_digits, current_digit, fps):
    """
    Draw user interface panel on frame
    """
    # Display info panel
    cv2.rectangle(frame, (10, 10), (280, 150), (30, 30, 30), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(detected_digits)}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(frame, f"Tracked: {len(processed_digits)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    cv2.putText(frame, f"Digit: {current_digit}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 252, 3), 2)

########################################
#           DRAW DETECTIONS FUNCTION
########################################
def draw_detection_boxes(frame, detections, predictions):
    """
    Draw bounding boxes and predictions on frame
    """
    for detection, (number, conf_num) in zip(detections, predictions):
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        
        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{number} ({conf_num:.2f})", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

########################################
#           PROCESS DETECTIONS FUNCTION
########################################
def process_detections(detections, frame, cnn_model, transform, current_time):
    """
    Process all detections and return recognized digits
    """
    global processed_digits, current_digit, counter
    
    detected_digits = []
    predictions = []
    
    for detection in detections:
        # Check if this is a new digit
        if not is_new_digit(detection, current_time):
            continue
        
        # Extract ROI
        roi = extract_roi(frame, detection)
        if roi is None:
            continue
        
        # Predict digit using CNN
        number, conf_num = predict_digit_with_cnn(roi, cnn_model, transform)
        
        if number != "none" and conf_num > 0.7:
            detected_digits.append((number, conf_num))
            predictions.append((number, conf_num))
            
            # Update current digit
            current_digit = number
            
            # Mark as processed
            key = f"{detection['center_x']}_{detection['center_y']}"
            processed_digits[key] = current_time
            
            print(f"🔢 Detected: {number} (Conf: {conf_num:.2f})")
            counter += 1
    
    return detected_digits, predictions

########################################
#           MAIN FUNCTION
########################################
def digit_recognition_loop():
    """Main function to run the digit recognition system"""
    global counter, current_digit, total_frames_processed
    
    # Initialize components
    yolo, cnn_model = load_models()
    transform = get_image_transforms()
    cap = initialize_camera()
    
    print("📹 Starting detection loop...")
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Unable to read frame from camera.")
                break
            
            # Update total frames counter
            total_frames_processed += 1
            
            # Skip frames to improve real-time performance
            if counter % frame_skip != 0:
                counter += 1
                continue
            
            current_time = time.time()
            
            # Run YOLO detection
            detections = run_yolo_detection(yolo, frame)
            
            # Process detections
            detected_digits, predictions = process_detections(
                detections, frame, cnn_model, transform, current_time
            )
            
            # Calculate FPS
            fps = 1 / (time.time() - last_time)
            last_time = time.time()
            
            # Draw UI and detections
            draw_ui_panel(frame, detected_digits, current_digit, fps)
            draw_detection_boxes(frame, detections, predictions)
            
            # Display frame
            cv2.imshow("Digit Detection (Optimized)", frame)
            
            # Handle user input
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                processed_digits.clear()
                print("🔄 Cleared memory")
            
            counter += 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cleanup and print statistics
        cap.release()
        cv2.destroyAllWindows()
        print_exit_statistics()

########################################
#           PROGRAM ENTRY POINT
########################################
