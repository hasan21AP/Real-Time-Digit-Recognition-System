import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model.model import RecognizeNumbersModel

# إعداد البيانات
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# مسار بيانات التحقق أو الاختبار
test_data_dir = "data/data_unified"   # عدل المسار حسب مجلدك
test_dataset = ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# تحميل النموذج
model_path = "weights/kaggle_printed_digits.pth"
model = RecognizeNumbersModel()
model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True))
model.eval()

# التقييم
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ CNN Model Accuracy: {accuracy:.2f}%  ({correct}/{total})")
