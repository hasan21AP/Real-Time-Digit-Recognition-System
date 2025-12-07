
import sys

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5n_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.7
ANDROID_SRC = "http://192.168.0.33:4747/video?fps=60"
LAPTOP_CAMERA_SRC = 0
CAMERA_SRC = LAPTOP_CAMERA_SRC  


sys.path.append("yolov5")

# torch.set_num_threads(1)
# cv2.setNumThreads(1)