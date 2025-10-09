import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.model import RecognizeNumbersModel

# Loading Model

model = RecognizeNumbersModel()
model.load_state_dict(torch.load("kaggle_printed_digits.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

lap_cam = 0
android_cam = "http://192.168.0.33:4747/video?fps=30"


# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # قناة وحدة
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: 1 - x),  # نقلب الألوان: أسود ↔ أبيض
    transforms.Normalize((0.5,), (0.5,))
])  

# Open Camera
cap = cv2.VideoCapture(android_cam)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Cut the frame to a square
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = 150
    roi = frame[cy-size:cy+size, cx-size:cx+size]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255,  cv2.THRESH_OTSU)
    # Image Processing
    img_pil = Image.fromarray(thresh)
    input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        labels = ["0","1","2","3","4","5","6","7","8","9","none"]
        predicted_label = labels[predicted_class]
    
    # Display the predicted number
    if confidence >= 0.9:  # Only show if confidence is high
        cv2.putText(frame, f"Predicted: {predicted_label} Confidence: {confidence:.2f}", 
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No digit detected", 
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (255, 0, 0), 2)
        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()