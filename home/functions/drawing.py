import cv2
from ..constants import global_variables as glv



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
    cv2.putText(frame, f"Tracked: {len(glv.processed_digits)}", (20, 85),
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