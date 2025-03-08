import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Swish activation function as a class
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(Swish())

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),
            MBConvBlock(16, 24, expand_ratio=2, stride=2),
            MBConvBlock(24, 24, expand_ratio=2, stride=1),
            MBConvBlock(24, 40, expand_ratio=2, stride=2),
            MBConvBlock(40, 40, expand_ratio=2, stride=1),
            MBConvBlock(40, 80, expand_ratio=4, stride=2),
            MBConvBlock(80, 80, expand_ratio=4, stride=1),
            MBConvBlock(80, 80, expand_ratio=4, stride=1),
            MBConvBlock(80, 112, expand_ratio=4, stride=1),
            MBConvBlock(112, 112, expand_ratio=8, stride=1),
            MBConvBlock(112, 192, expand_ratio=8, stride=2),
            MBConvBlock(192, 192, expand_ratio=8, stride=1),
            MBConvBlock(192, 192, expand_ratio=8, stride=1),
            MBConvBlock(192, 320, expand_ratio=8, stride=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
