from ..functions import tracking
from ..functions import models
from ..functions import cnn_prediction
from constants import global_variables as glv

########################################
#           PROCESS DETECTIONS FUNCTION
########################################
def process_detections(detections, frame, cnn_model, transform, current_time):
    """
    Process all detections and return recognized digits
    """
    
    detected_digits = []
    predictions = []
    
    for detection in detections:
        # Check if this is a new digit
        if not tracking.is_new_digit(detection, current_time):
            continue
        
        # Extract ROI
        roi = models.extract_roi(frame, detection)
        if roi is None:
            continue
        
        # Predict digit using CNN
        number, conf_num = cnn_prediction.predict_digit_with_cnn(roi, cnn_model, transform)
        
        if number != "none" and conf_num > 0.7:
            detected_digits.append((number, conf_num))
            predictions.append((number, conf_num))
            
            # Update current digit
            glv.current_digit = number
            
            # Mark as processed
            key = f"{detection['center_x']}_{detection['center_y']}"
            glv.processed_digits[key] = current_time
            
            print(f"🔢 Detected: {number} (Conf: {conf_num:.2f})")
            counter += 1
    
    return detected_digits, predictions

