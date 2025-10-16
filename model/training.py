from .model import RecognizeNumbersModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os 
from torch.utils.data import DataLoader, random_split


path = "data_unified_64"


data_dir = os.path.join(path)

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.Grayscale(),  # Make sure images are single channel
    transforms.Resize((64, 64)), # Resize to 28x28
    transforms.RandomRotation(20), # Simple Rotation +- 20 degrees
    transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.5, 1.2)),  # Random translation 15%
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Random Perspective
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # Random brightness/contrast change
    transforms.ToTensor(), # Convert image to tensor [0,1]
    transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1,1]
])
data_set = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset into training and testing sets (80-20 split)
train_size = int(0.8 * len(data_set))
test_size  = len(data_set) - train_size
train_dataset, test_dataset = random_split(data_set, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= 64, shuffle=False)

print(f"📊 Dataset size: {len(data_set)} images")
print(f"🧩 Train: {len(train_dataset)}, Test: {len(test_dataset)}")


model = RecognizeNumbersModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



    
# Model training
def model_training(*, epochs=5):
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
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
        accuracy = 100 * correct / total
        if accuracy >= best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "weights/kaggle_printed_digits.pth")
            print(f"✅ Model improved! Saved with accuracy = {best_acc:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")
    print("🏁 Training complete")
    print("🎉 Training finished. Best accuracy:", best_acc)