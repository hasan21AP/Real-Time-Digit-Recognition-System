import torch
from .model import RecognizeNumbersModel
from PIL import Image
import torchvision.transforms as transforms


def load_model():
    model = RecognizeNumbersModel()
    model.load_state_dict(torch.load("weights/kaggle_printed_digits.pth", weights_only=True))
    model.eval()   # وضع inference


    # نفس التحويل اللي استعملناه
    transform = transforms.Compose([
        transforms.Grayscale(),        # قناة وحدة
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # نقلب الألوان: أسود ↔ أبيض
        transforms.Normalize((0.5,), (0.5,))
    ])

    # target_number = random.randrange(0, 10)
    for number in range(0, 10):
        img_path = f"model/data/assets/{number}.png"
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0)  # إضافة batch dimension

        with torch.no_grad():
            output = model(image)
            predicted = torch.argmax(output, dim=1).item()

        print(f"Actual number: {number} (Predicted number: {predicted})")