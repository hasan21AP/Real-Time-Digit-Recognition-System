from ..constants import global_variables as glv
import cv2



# --- Tracker helers (CSRT preferred, MOSSE fallback) ---
def create_tracker(prefer="CSRT"):
    """Create and return an OpenCV tracker instance with legacy fallback."""
    if prefer.upper() == "CSRT":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
    # Fallback to MOSSE (faster but less accurate)
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()
    raise RuntimeError("No supported tracker found (CSRT/MOSSE).")

def iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union





def clamp_box(b, w, h, pad=0):
    x1 = max(0, min(w-1, b[0]-pad)); y1 = max(0, min(h-1, b[1]-pad))
    x2 = max(0, min(w-1, b[2]+pad)); y2 = max(0, min(h-1, b[3]+pad))
    if x2 <= x1 or y2 <= y1: return None
    return (x1, y1, x2, y2)

def init_flow_keypoints(gray, bbox):
    x1,y1,x2,y2 = bbox
    roi = gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=glv.FLOW_FEATURES, qualityLevel=0.01, minDistance=5)
    if pts is None: return None
    # shift to image coords
    pts[:,0,0] += x1; pts[:,0,1] += y1
    return pts



########################################
#           DIGIT TRACKING FUNCTION
########################################
def is_new_digit(detection, current_time):
    """
    Check if detection is a new digit based on spatial-temporal filtering
    """
    
    key = f"{detection['center_x']}_{detection['center_y']}"
    
    # Remove outdated detections
    glv.processed_digits = {
        d: t for d, t in glv.processed_digits.items()
        if current_time - t < glv.DIGIT_MEMORY_TIME
    }
    
    return key not in glv.processed_digits