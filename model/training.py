import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms





# تحويل الصورة إلى Tensor وتطبيعها بين 0 و 1
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# تحميل بيانات MNIST للتدريب والاختبار
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolution 1: 1 قناة → 16 خرائط ميزات
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Convolution 2: 16 → 32 خرائط ميزات
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Convolution 3: 32 → 64 خرائط ميزات
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Convolution 3: 64 → 128 خرائط ميزات
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # MaxPooling 2x2
        self.pool = nn.MaxPool2d(2, 2)
        # Fully Connected Layer
        self.fc1 = nn.Linear(in_features=128 * 1 * 1, out_features=128)  # بعد Pooling
        self.fc2 = nn.Linear(128, 10)          # 10 أرقام
        self.dropout = nn.Dropout(0.5)

        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv3(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(F.relu(self.conv4(x)))  # Conv2 + ReLU + Pooling
        x = x.view(-1, 128 * 1 * 1)           # Flatten
        x = F.relu(self.fc1(x))               # Fully Connected
        x = self.dropout(x) 
        x = self.fc2(x)                       # Output logits
        return x

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



    
# تدريب المودل
model.train()
def model_training(*, epochs=5):
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")

        torch.save(model.state_dict(), "mnist_cnn.pth")