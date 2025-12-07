##### Digit Recognition In Real Time (Optimized Version for CPU) #####
# Author: Hasan Game (optimized by ChatGPT)
# Description:
# Lightweight version of the YOLO + CNN digit recognition system
# optimized for CPU inference with reduced latency and power consumption.

from home.functions import models
import time
from home.constants import global_variables as glv
from home.functions import cnn_prediction
from home.functions import tracking
from home.functions import drawing
from home.functions import perfomance_monitoring as perf
import cv2
########################################
#           MAIN FUNCTION
########################################
def digit_recognition_loop():
    """Main function to run the digit recognition system"""
    
    # Initialize components
    yolo, cnn_model = models.load_models()
    transform = models.get_image_transforms()
    cap = models.initialize_camera()
    
    print("📹 Starting detection loop...")
    h, w = None, None
    last_time = time.time()
    glv.tracker = None
    glv.track_bbox = None
    glv.track_lost = 0
    glv.frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Unable to read frame from camera.")
                break

            glv.total_frames_processed += 1
            glv.frame_idx += 1

                        # per-frame drawing gate
            detections_for_draw = []
            predictions_for_draw = []
            draw_box = False

            use_yolo = (glv.frame_idx % glv.YOLO_EVERY_N == 1) or (glv.tracker is None)

            # (A) Try tracker first when not running YOLO this frame
            if glv.tracker is not None and not use_yolo:
                ok, box = glv.tracker.update(frame)  # (x, y, w, h)
                if ok:
                    x, y, w, h = box
                    glv.track_bbox = (int(x), int(y), int(x + w), int(y + h))
                    glv.track_lost = 0
                    draw_box = True

                    # classify ROI while tracking
                    roi = models.extract_roi(frame, {"x1": glv.track_bbox[0], "y1": glv.track_bbox[1], "x2": glv.track_bbox[2], "y2": glv.track_bbox[3]})
                    if roi is not None:
                        number, conf_num = cnn_prediction.predict_digit_with_cnn(roi, cnn_model, transform)
                        if number != "none":
                            glv.current_digit = number
                            predictions_for_draw.append((number, conf_num))
                    detections_for_draw.append({
                        "x1": glv.track_bbox[0], "y1": glv.track_bbox[1],
                        "x2": glv.track_bbox[2], "y2": glv.track_bbox[3],
                        "confidence": 1.0
                    })
                else:
                    # lost this frame
                    glv.track_lost += 1
                    if glv.track_lost > max(0, glv.GHOST_FRAMES):
                        glv.tracker = None
                        glv.track_bbox = None

            # (B) Run YOLO periodically or if no active tracker
            if use_yolo:
                dets = models.run_yolo_detection(yolo, frame)
                best = None
                if dets:
                    best = max(dets, key=lambda d: d["confidence"])
                    if glv.track_bbox is not None:
                        i = tracking.iou(glv.track_bbox, (best["x1"], best["y1"], best["x2"], best["y2"]))
                        if i < glv.IOU_MATCH_THR and glv.tracker is not None:
                            best = None  # keep current track if YOLO disagrees too much

                if best is not None:
                    # (re)init tracker on YOLO box
                    glv.tracker = tracking.create_tracker(glv.TRACK_PREFER)
                    x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
                    w, h = x2 - x1, y2 - y1
                    glv.tracker.init(frame, (x1, y1, w, h))
                    glv.track_bbox = (x1, y1, x2, y2)
                    glv.track_lost = 0
                    draw_box = True

                    # classify selected ROI
                    roi = models.extract_roi(frame, best)
                    if roi is not None:
                        number, conf_num = cnn_prediction.predict_digit_with_cnn(roi, cnn_model, transform)
                        if number != "none":
                            glv.current_digit = number
                            predictions_for_draw.append((number, conf_num))
                    detections_for_draw.append({**best, "confidence": best.get("confidence", 1.0)})
                else:
                    # no YOLO detection this cycle
                    if glv.tracker is None:
                        glv.track_bbox = None  # ensure nothing is drawn

            # (C) FPS + drawing
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now

            drawing.draw_ui_panel(frame, predictions_for_draw, glv.current_digit, fps)

            # only draw a box when we actually have one this frame
            if draw_box and detections_for_draw and predictions_for_draw:
                drawing.draw_detection_boxes(frame, detections_for_draw, predictions_for_draw)
            elif draw_box and detections_for_draw:
                # draw plain box without label if no prediction this frame
                for d in detections_for_draw:
                    cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 200, 255), 2)


            cv2.imshow("Digit Detection (Optimized + Tracking)", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                glv.processed_digits.clear()
                glv.tracker = None
                glv.track_bbox = None
                glv.track_lost = 0
                print("🔄 Cleared memory and reset tracker")

            
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        # Cleanup and print statistics
        cap.release()
        cv2.destroyAllWindows()
        perf.print_exit_statistics()

########################################
#           PROGRAM ENTRY POINT
########################################