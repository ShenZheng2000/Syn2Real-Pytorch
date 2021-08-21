# Purpose: Use channel attention layers with skip connections

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Depthwise Separable Convolution
class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# Pixel attention
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# Channel Residual Attention Layer
class CRALayer(nn.Module):
    def __init__(self, channel, reduction): # channel = 32, reduction = 16
        super(CRALayer, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature Channel Rescale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
        )
        # 1 X 1 Convolution inside Skip Connection
        self.conv_1_1 = nn.Conv2d(channel, channel, 1, padding=0, bias=False)
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        res = self.conv_1_1(y)
        y = self.conv_du(y)
        y += res
        y = self.sigmoid(y)
        return x * y

# Pixel Residual Attention Layer
class PRALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(PRALayer, self).__init__()
        # Feature Channel Rescale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
        )
        # 1 X 1 Convolution inside Skip Connection
        self.conv_1_1 = nn.Conv2d(channel, channel, 1, padding=0, bias=False)
        # Sigmoid Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv_1_1(x)
        y = self.conv_du(x)
        y += res
        y = self.sigmoid(y)
        return x * y