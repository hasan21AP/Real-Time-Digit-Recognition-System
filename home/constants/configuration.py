import cv2 
import sys
import torch

########################################
#           CONFIGURATION
########################################
YOLO_MODEL_PATH = "weights/yolov5s_trained_v1.pt"
CNN_MODEL_PATH = "weights/kaggle_printed_digits.pth"
CONF_THRESHOLD = 0.85
CAMERA_SRC = 0  # laptop webcam by default (use phone stream if needed)
ANDROID_SRC = "http://192.168.0.33:4747/video"


sys.path.append("yolov5")

torch.set_num_threads(1)
cv2.setNumThreads(1)