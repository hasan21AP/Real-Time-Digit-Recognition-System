import torch.nn as nn



class RecognizeNumbersModel(nn.Module):
    def __init__(self):
        super(RecognizeNumbersModel, self).__init__()
        # Convolution 1: 1 -> 16 Features Maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride= 2)
        # Convolution 2: 16 -> 32 Features Maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride= 2)
        # Convolution 3: 32 -> 64 Features Maps
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride= 2)
        # Convolution 4: 64 -> 128 Features Maps
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride= 2)
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=64)  
        self.fc2 = nn.Linear(in_features=64, out_features=11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv + ReLU
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        