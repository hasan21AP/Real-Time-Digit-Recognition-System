from PIL import Image
import torch
from ..constants import global_variables as glv
import cv2


########################################
#           CNN PREDICTION FUNCTION
########################################
def predict_digit_with_cnn(roi, cnn_model, transform):
    """
    Predict digit from ROI using CNN model
    """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)
        
        with torch.no_grad():
            output = cnn_model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            number = glv.labels[predicted.item()]
            conf_num = confidence.item()
        
        return number, conf_num
    except Exception as e:
        print(f"❌ CNN prediction error: {e}")
        return "none", 0.0