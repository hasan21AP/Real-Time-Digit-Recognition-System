import cv2
from model.training import SimpleCNN
import torch
from PIL import Image
import torchvision.transforms as transforms
import easyocr

reader = easyocr.Reader(['en'])

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
    
    results = reader.readtext(frame)
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            confidence = confidence.item()
            predicted = predicted_class.item()

        if confidence > 0.9:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted} ({confidence*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()