import torch.nn as nn
import torch
import math
from .conv import DSC

# Modified Res2Net bottle neck
class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 baseWidth,
                 cardinality,
                 stride=1,
                 downsample=None,
                 scale=4,
                 stype='normal',
                 conv = 'tc'): # dsc or tc # TODO: why dc does not reduce params?
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C * scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(D * C * scale)
        # self.SE = SEBlock(inplanes,C)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            if conv == 'tc':
                convs.append(nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False))
            else:
                convs.append(DSC(D * C, D * C))
            bns.append(nn.BatchNorm2d(D * C))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D * C * scale, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width = D * C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # pdb.set_trace()
        out += residual

        return torch.cat([x, out], 1)
