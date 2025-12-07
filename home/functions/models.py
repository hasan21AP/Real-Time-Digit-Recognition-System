import torch
from model.model import RecognizeNumbersModel 
from ..constants import configuration as config
import torchvision.transforms as transforms
import cv2
import sys

########################################
#           MODEL LOADING FUNCTION
########################################
def load_models():
    """Load YOLO and CNN models optimized for CPU"""
    print("🚀 Loading models (optimized for CPU)...")
    
    # Load lightweight YOLOv5n model
    yolo = torch.hub.load("yolov5", "custom", path=config.YOLO_MODEL_PATH, source="local", device="cpu")
    
    # Disable FP16 for CPU (it slows down)
    if torch.cuda.is_available():
        yolo.half()
    
    # Load CNN classifier
    model = RecognizeNumbersModel()
    model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location="cpu", weights_only=True))
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
    cap = cv2.VideoCapture(config.CAMERA_SRC)
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
        if conf < config.CONF_THRESHOLD:
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