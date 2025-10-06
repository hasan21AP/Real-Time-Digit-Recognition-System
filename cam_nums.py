import cv2
from model.training import SimpleCNN
import torch
from PIL import Image
import torchvision.transforms as transforms

# Loading Model

model = SimpleCNN()
model.load_state_dict(torch.load("kaggle_printed_numbers.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()


# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # قناة وحدة
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: 1 - x),  # نقلب الألوان: أسود ↔ أبيض
    transforms.Normalize((0.5,), (0.5,))
])  

# Open Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detecting the number
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussionBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            continue

        roi = gray[y:y+h, x:x+w]
        img_pil = Image.fromarray(roi)
        input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        predicted = predicted_class.item()

    if confidence > 0.95:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted} ({confidence*100:.1f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()