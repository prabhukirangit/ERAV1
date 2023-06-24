import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):
    def __init__(self,norm='BatchNorm2d',GroupSize=1):
        super(Net, self).__init__()
        self.dropout_value=0.05
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=2, bias=False),
            nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=2, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
			nn.BatchNorm2d(12) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,12) if norm=='GroupNorm' else nn.GroupNorm(1,12),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)


        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
			nn.BatchNorm2d(12) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,12) if norm=='GroupNorm' else nn.GroupNorm(1,12),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # OUTPUT BLOCK
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
			nn.BatchNorm2d(16) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,16) if norm=='GroupNorm' else nn.GroupNorm(1,16),
            nn.ReLU()
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU()
        )

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
			nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU()
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = x + self.convblock5(x)
        x = x + self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
                               
def model_summary(model,input_size):
    return summary(model, input_size)