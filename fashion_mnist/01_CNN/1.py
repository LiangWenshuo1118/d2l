import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1，stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) # padding默认值是0
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
             X = self.conv3(X)
             Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=3), # 输入1*28*28 > 输出16 * 28 * 28
    nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2) # 输入16 * 28 * 28 > 输出16 * 14 * 14
    )

# 输入16 * 14 * 14 > 输出 16 * 14 * 14 
b2 = nn.Sequential(*resnet_block(16, 16, 2, first_block=True)) 

# 输入16 * 14 * 14 > 输出 32 * 7 * 7 
b3 = nn.Sequential(*resnet_block(16, 32, 2))

net = nn.Sequential(b1, b2,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Linear(32, 10)
    )
