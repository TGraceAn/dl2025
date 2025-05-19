import torch
import torch.nn as nn


# implement CNN
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # assume input is 28x28
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        # self.dropout = nn.Dropout(p=0.5) # Optional dropout layer

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7) # flatten
        x = self.relu(self.fc1(x))

        # x = self.dropout(x) # Optional dropout layer

        x = self.fc2(x)
        return x
    
    