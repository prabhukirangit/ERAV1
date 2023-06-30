import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class Net(nn.Module):
    def __init__(self,norm='BatchNorm2d',GroupSize=1):
        super(Net, self).__init__()
        self.dropout_value=0.05

         # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=2, bias=False),
            nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False),
			      nn.BatchNorm2d(64) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, bias=False,dilation=2,stride=2),
			      nn.BatchNorm2d(64) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,12) if norm=='GroupNorm' else nn.GroupNorm(1,12),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )


        # CONVOLUTION BLOCK 2
        self.depthwiseConvlution = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,groups=64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
			      nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # self.depthwise = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)
        # self.pointwise = nn.Conv2d(3, 64, kernel_size=1)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
			      nn.BatchNorm2d(48) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,dilation=2),
			      nn.BatchNorm2d(64) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
			      nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
			      nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,dilation=2),
			      nn.BatchNorm2d(64) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        # CONVOLUTION BLOCK 4
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
			      nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
			      nn.BatchNorm2d(32) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False,dilation=2),
			      nn.BatchNorm2d(64) if norm=='BatchNorm2d' else nn.GroupNorm(GroupSize,32) if norm=='GroupNorm' else nn.GroupNorm(1,32),
            nn.ReLU(),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.depthwiseConvlution(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock12(x)
        x = self.gap(x)
        x = self.convblock13(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def model_summary(model,input_size):
    return summary(model, input_size)