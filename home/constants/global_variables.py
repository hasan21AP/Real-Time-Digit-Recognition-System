import time

########################################
#           GLOBAL VARIABLES
########################################
counter = 0
labels = [str(i) for i in range(10)] + ["none"]
processed_digits = {}
DIGIT_MEMORY_TIME = 0.02  # seconds
current_digit = None
frame_skip = 0  # process every 2nd frame to reduce load

# Statistics variables
start_time = time.time()
total_frames_processed = 0
fps_history = []

# --- Tracking constants ---
IOU_THRESH = 0.40        # association threshold between old track and new detection
MAX_LOST = 20            # keep drawing box this many frames without a detection
REDETECT_EVERY = 4       # run YOLO every N frames (lower = more stable, higher = faster)
FLOW_FEATURES = 60       # Shi-Tomasi features inside ROI for optical flow
SMOOTH_ALPHA = 0.6       # box smoothing (0..1), higher = smoother
# ---- Tracker configuration ----
# --- Tracking config ---
TRACK_PREFER = "CSRT"   # "CSRT" for accuracy, "MOSSE" for speed
YOLO_EVERY_N = 3        # run YOLO every N frames
IOU_MATCH_THR = 0.35    # match YOLO detection to current track
MAX_TRACK_LOST = 12     # keep drawing for this many frames after lost
GHOST_FRAMES = 0   # draw nothing when tracker is lost (no ghost box)

# --- Tracking state ---
tracker = None
track_bbox = None       # (x1,y1,x2,y2)
track_lost = 0
frame_idx = 0