from PIL import Image
import torch
from torchvision.transforms import transforms
from ..constants import configuration as config
from ..constants import global_variables as glv


########################################
#           CNN PREDICTION FUNCTION
########################################
def predict_digit_with_cnn(roi, cnn_model, transform):
    """
    Run CNN digit prediction on a single ROI.
    roi: numpy array (BGR) from OpenCV.
    cnn_model: CNN model on CPU.
    transform: torchvision.transforms.Compose.
    """
    try:
        if roi is None or roi.size == 0:
            return "none", 0.0

        # Apply transforms (should return a tensor [C,H,W])
        img = transform(roi)

        # Safety: if transform returns a PIL image, convert to tensor
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)

        # Add batch dimension -> [1, C, H, W]
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.to(torch.device("cpu"))

        cnn_model.eval()
        with torch.no_grad():
            outputs = cnn_model(img)            # [1, num_classes]
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        digit_idx = int(pred_idx.item())
        confidence = float(conf.item())
        
        # reject low-confidence predictions
        if confidence < config.CONF_THRESHOLD:
            return "none", confidence

        # map unknown/background class to "none"
        if digit_idx == config.UNKNOWN_CLASS_INDEX or digit_idx < 0 or digit_idx > 9:
            return "none", confidence

        return str(digit_idx), confidence

    except Exception as e:
        print(f"❌ CNN prediction internal error: {e}")
        return "none", 0.0