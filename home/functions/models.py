import torch
from model.model import RecognizeNumbersModel 
from ..constants import configuration as config
import torchvision.transforms as transforms
import cv2
import sys
from pathlib import Path
import numpy as np

# ===== add yolov5 path =====
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2] / "yolov5"   # ..\..\yolov5

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes



# ===== choose device (GPU if available) =====
try:
    DEVICE = select_device("0" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"⚠️ Failed to use CUDA, falling back to CPU: {e}", flush=True)
    DEVICE = select_device("cpu")

print(f"🔥 Using device: {DEVICE}", flush=True)

########################################
#           MODEL LOADING FUNCTION
########################################
def load_models():
    """
    Load YOLOv5 model (on GPU if available) + CNN model on CPU.
    """

    # ---- YOLO model ----
    print("🔄 Loading YOLO model...", flush=True)
    yolo_model = DetectMultiBackend(
        config.YOLO_MODEL_PATH,                        # "weights/yolov5n_trained_v1.pt"
        device=DEVICE,
        data=str(ROOT / "data" / "coco128.yaml"),
        fp16=(DEVICE.type != "cpu"),               # use fp16 on GPU for speed
    )
    print(f"✅ YOLO model loaded on {DEVICE}, fp16={yolo_model.fp16}", flush=True)

    # ---- CNN model ----
    print("🔄 Loading CNN model...", flush=True)
    cnn_model = RecognizeNumbersModel()
    cnn_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location=DEVICE))
    cnn_model.eval()
    print(f"✅ CNN model loaded on {DEVICE}.", flush=True)

    return yolo_model, cnn_model

########################################
#           IMAGE TRANSFORMS FUNCTION
########################################
def get_image_transforms():
    """Define image transformations for CNN"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((128, 128)),  # smaller input improves speed
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
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
    Run YOLOv5 inference (GPU if available, CPU otherwise).
    Returns list of dicts: {x1, y1, x2, y2, confidence}
    """
    img0 = frame.copy()

    # ---- preprocess ----
    img = letterbox(img0, config.YOLO_IMG_SIZE, stride=yolo_model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(yolo_model.device)
    im = im.half() if yolo_model.fp16 else im.float()
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # ---- inference ----
    with torch.no_grad():
        pred = yolo_model(im)

    # ---- NMS ----
    pred = non_max_suppression(
        pred,
        config.CONF_THRESHOLD,
        config.IOU_THRESHOLD,
        max_det=20,
    )

    detections = []
    for det in pred:
        if len(det):
            # scale boxes back to original frame size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": float(conf),
                    }
                )

    return detections

########################################
#           ROI EXTRACTION FUNCTION
########################################
def extract_roi(frame, detection):
    """
    Extract Region of Interest from frame based on detection
    """
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    h, w, _ = frame.shape
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size != 0 else None