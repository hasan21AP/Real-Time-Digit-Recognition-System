import cv2
from model.training import SimpleCNN
import torch
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")  # Loading a pretrained model (YOLOv8n)

# Loading Model

model = SimpleCNN()
model.load_state_dict(torch.load("mnist_finetuned.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()


# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # قناة وحدة
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    # transforms.Lambda(lambda x: 1 - x),  # نقلب الألوان: أسود ↔ أبيض
    transforms.Normalize((0.5,), (0.5,))
])  

# Open Camera
cap = cv2.VideoCapture('http://192.168.0.33:4747/video?fps=30')  # Change to 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (320, 320))
    results = yolo_model(frame, verbose=False)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]

        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 50 or w > 300 or h > 300:
            continue


        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        # Preprocessing the ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_pil = Image.fromarray(thresh)
        input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted = predicted_class.item()
        
        # Display the predicted number
        label = f"{predicted} ({confidence*100:.1f}%)"
        if confidence > 0.95:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()