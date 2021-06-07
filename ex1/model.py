import torch
from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding_mode='reflect')
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding_mode='reflect')
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding_mode='reflect')
        self.maxp3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(self.get_dims(), 600)
        self.fc2 = nn.Linear(600, 250)
        self.fc3 = nn.Linear(250, 10)
        
        self.dropout_conv = nn.Dropout2d(p=0.33)
        self.dropout_fc = nn.Dropout(p=0.33)
    
    def get_dims(self):
        x_dummy = torch.zeros_like(torch.empty(64, 1, 28, 28))
        x_dummy = self.conv1(x_dummy)
        x_dummy = self.maxp1(x_dummy)
        x_dummy = self.conv2(x_dummy)
        x_dummy = self.maxp2(x_dummy)
        x_dummy = self.conv3(x_dummy)
        x_dummy = self.maxp3(x_dummy)
        return x_dummy.view(x_dummy.size(0), -1).shape[1]
    
    def forward(self, x):
        # Conv 1
        x = F.relu(self.conv1(x))
        x = self.maxp1(x)
        x = self.dropout_conv(x)

        # Conv 2
        x = F.relu(self.conv2(x))
        x = self.maxp2(x)
        x = self.dropout_conv(x)

        # Conv 3
        x = F.relu(self.conv3(x))
        x = self.maxp3(x)
        x = self.dropout_conv(x)

        # Fully-connected
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        x = self.dropout_fc(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
