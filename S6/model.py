import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.1)

        self.conv6 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 10, 1)
        self.gap1 = nn.AvgPool2d(7)


    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(F.relu(self.bn5(self.conv5(F.relu(self.bn4(self.conv4(self.bn3(F.relu(self.conv3(x)))))))))))
        x = self.gap1(self.conv7(F.relu(self.bn6(self.conv6(x)))))
        x = x.view(-1, 10)
        return F.log_softmax(x)

def model_summary(model,input_size):
    return summary(model, input_size)
