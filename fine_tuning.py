import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model.training import SimpleCNN
import kagglehub
import os 


# Download Printed Digits dataset automatically
print("📦 Downloading Printed Digits Dataset from Kaggle...")
path = kagglehub.dataset_download("kshitijdhama/printed-digits-dataset")
print("✅ Dataset downloaded successfully at:", path)

# Load the Dataset

transform = transforms.Compose([
    transforms.Grayscale(),                        # Convert image to single channel
    transforms.Resize((28, 28)),                   # Resize to 28x28 pixels
    transforms.RandomRotation(20),                 # Random rotation within ±20 degrees
    transforms.RandomAffine(0, translate=(0.2, 0.2)),  # Random translation in both axes
    transforms.ColorJitter(brightness=0.4, contrast=0.4),  # Random brightness/contrast change
    transforms.RandomHorizontalFlip(),             # Random horizontal flip
    transforms.ToTensor(),                         # Convert image to tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))           # Normalize to range [-1,1]
])

train_dataset = datasets.ImageFolder(root=os.path.join(path, "assets"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load Pre-trained Model
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location='cuda', weights_only=True))
model.eval()

def fine_tune_model(* ,epochs=5):
    
    # Unfreeze all layers except the last fully connected layer
    for name, param in model.named_parameters():
        if "conv4" in name or "fc" in name: # Unfreeze last conv layer and fc layers
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # Replace the last fully connected layer with a new one
    model.fc2 = nn.Linear(128, 10)

    # Traning Mode
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "mnist_finetuned.pth")
    print("✅ Fine-tuning done and model saved.")