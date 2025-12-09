
import sys

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5n_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
ANDROID_SRC = "http://192.168.0.33:4747/video?fps=60"
LAPTOP_CAMERA_SRC = 0
CAMERA_SRC = ANDROID_SRC  # Change to LAPTOP_CAMERA_SRC for laptop camera  
CONF_THRESHOLD  = 0.8
IOU_THRESHOLD   = 0.45
YOLO_IMG_SIZE   = 640
UNKNOWN_CLASS_INDEX = 10


sys.path.append("yolov5")

# torch.set_num_threads(1)
# cv2.setNumThreads(1)