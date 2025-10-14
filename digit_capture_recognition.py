import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.model import RecognizeNumbersModel
from ultralytics import YOLO
import time, os



# Loading Models
yolo = YOLO("yolo/runs/detect/yolo_digits/weights/best.pt")

model = RecognizeNumbersModel()
model.load_state_dict(torch.load("weights/kaggle_printed_digits.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

LAP_CAM = 0
ANDROID_CAM = "http://192.168.0.33:4747/video?fps=30"
ANDROID_CAM_MADAR = "http://10.35.92.27:4747/video?fps=30"


# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # One channel
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])  

# Open Camera
cap = cv2.VideoCapture(ANDROID_CAM)
os.makedirs("captures", exist_ok=True)

last_capture_time = 0
min_interval = 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = yolo(frame, verbose=False )
    detections = results[0].boxes
    
    if len(detections) > 0:
        for box in detections:
            conf = float(box.conf[0])
            if conf > 0.6:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if time.time() - last_capture_time > min_interval:
                    roi = frame[y1:y2, x1:x2]
                    filename = f"captures/{int(time.time())}.png"
                    cv2.imwrite(filename, roi)
                    print(f"📸 Screenshot {conf:.2f} And saved in {filename}")
                    last_capture_time = time.time()

                    # Analyze the captured ROI
                    roi = cv2.resize(roi, (128, 128))
                    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB, ))
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU)
                    img = Image.fromarray(thresh)
                    img = transform(img).unsqueeze(0)
        
                    with torch.no_grad():
                        output = model(img)
                        probs = torch.softmax(output, dim=1)
                        confidence, predicted = torch.max(probs, dim=1)
                        
                    labels = ["0","1","2","3","4","5","6","7","8","9","none"]
                    number = labels[predicted.item()]
                    print(f"🔢 Predicted number: {number} (Confidence: {confidence.item():.2f})")
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
