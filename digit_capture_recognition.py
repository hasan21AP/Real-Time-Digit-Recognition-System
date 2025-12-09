##### Digit Recognition In Real Time (Simple Version - No Tracking) #####
# Author: Alhassan Abdulmalik
# Description:
# Clean and simple version of the YOLO + CNN digit recognition system.
# No tracker, no ghost frames. A box is drawn ONLY when we have detections
# in the *current* frame, so no stale boxes can remain.

import time
import cv2

from home.functions import models
from home.functions import cnn_prediction
from home.functions import drawing
from home.functions import perfomance_monitoring as perf
from home.constants import global_variables as glv


########################################
#           MAIN FUNCTION
########################################
def digit_recognition_loop():
    """Main function to run the digit recognition system (simple version)."""

    # ---------------- Init components ----------------
    yolo_model, cnn_model = models.load_models()
    transform = models.get_image_transforms()
    cap = models.initialize_camera()

    print("📹 Starting detection loop (simple, no tracking)...")

    # reset global stats (if you use them in perf module)
    glv.total_frames_processed = 0
    glv.current_digit = None

    last_time = time.time()

    try:
        while True:
            # -------------- read frame --------------
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Unable to read frame from camera.")
                break

            glv.total_frames_processed += 1

            # -------------- per-frame state --------------
            # these lists live ONLY for this frame
            detections_for_draw = []
            predictions_for_draw = []

            # -------------- YOLO detection (every frame) --------------
            dets = models.run_yolo_detection(yolo_model, frame)

            if dets:
                # you can keep all detections, or only the best one.
                # Here: choose the best detection by confidence.
                best = max(dets, key=lambda d: d.get("confidence", 0.0))

                # extract ROI for CNN classification
                roi = models.extract_roi(frame, best)
                if roi is not None:
                    number, conf_num = cnn_prediction.predict_digit_with_cnn(
                        roi, cnn_model, transform
                    )

                    if number != "none":
                        glv.current_digit = number
                        detections_for_draw.append(best)
                        predictions_for_draw.append((number, conf_num))

            # -------------- FPS calculation --------------
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now

            # -------------- UI panel --------------
            drawing.draw_ui_panel(
                frame,
                predictions_for_draw,
                glv.current_digit,
                fps,
            )

            # -------------- draw boxes --------------
            # IMPORTANT:
            # Boxes are drawn ONLY if we have detections THIS FRAME.
            # If dets == [] → detections_for_draw is empty → NOTHING is drawn.
            if detections_for_draw:
                drawing.draw_detection_boxes(
                    frame,
                    detections_for_draw,
                    predictions_for_draw,
                )

            # -------------- show frame --------------
            cv2.imshow("Digit Detection (Simple)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # clear processed digits history if you use it
                if hasattr(glv, "processed_digits"):
                    glv.processed_digits.clear()
                glv.current_digit = None
                print("🔄 Cleared memory (processed digits).")

    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user.")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # print exit statistics (if you use global counters there)
        try:
            perf.print_exit_statistics()
        except Exception as e:
            print(f"⚠️ Could not print statistics: {e}")
