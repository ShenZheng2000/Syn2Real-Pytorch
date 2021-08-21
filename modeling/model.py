from utils import *
from .Res2Net import *
from .ResNeSt import *


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, choice='1', dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate
        self.choice = choice

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.choice == '1':  # upsample
            return F.upsample_nearest(out, scale_factor=2)
        elif self.choice == '2':  # downsample
            return F.avg_pool2d(out, 2)
        elif self.choice == '3':  # same
            return out


class DeRain_v2(nn.Module):
    def __init__(self, split = False):
        super(DeRain_v2, self).__init__()
        self.split = split # ResNeST or Res2Net block
        self.baseWidth = 12  # 4#16
        self.cardinality = 8  # 8#16
        self.scale = 6  # 4#5
        self.stride = 1
        ############# Block1-scale 1.0  ##############
        self.conv_input = nn.Conv2d(3, 16, 3, 1, 1)
        self.dense_block1 = Bottle2neckX(16, 16, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block1s = Bottleneck(16, 16, stride=self.stride, cardinality=self.cardinality)

        ############# Block2-scale 0.50  ##############
        self.trans_block2 = TransitionBlock(32, 32, choice='2')
        self.dense_block2 = Bottle2neckX(32, 32, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block2s = Bottleneck(32, 32, stride=self.stride, cardinality=self.cardinality)
        self.trans_block2_o = TransitionBlock(64, 32, choice='3')

        ############# Block3-scale 0.250  ##############
        self.trans_block3 = TransitionBlock(32, 32, choice='2')
        self.dense_block3 = Bottle2neckX(32, 32, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block3s = Bottleneck(32, 32, stride=self.stride, cardinality=self.cardinality)
        self.trans_block3_o = TransitionBlock(64, 64, choice='3')

        ############# Block4-scale 0.25  ##############
        self.trans_block4 = TransitionBlock(64, 128, choice='2')
        self.dense_block4 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block4s = Bottleneck(128, 128, stride=self.stride, cardinality=self.cardinality)
        self.trans_block4_o = TransitionBlock(256, 128, choice='3')

        ############# Block5-scale 0.25  ##############
        self.dense_block5 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block5s = Bottleneck(128, 128, stride=self.stride, cardinality=self.cardinality)
        self.trans_block5_o = TransitionBlock(256, 128, choice='3')

        ############# Block6-scale 0.25  ##############
        self.dense_block6 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block6s = Bottleneck(128, 128, stride=self.stride, cardinality=self.cardinality)
        self.trans_block6_o = TransitionBlock(256, 128, choice='3')

        ############# Block7-scale 0.25  ############## 7--3 skip connection
        self.trans_block7 = TransitionBlock(32, 64)
        self.dense_block7 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block7s = Bottleneck(128, 128, stride=self.stride, cardinality=self.cardinality)
        self.trans_block7_o = TransitionBlock(256, 32, choice='3')

        ############# Block8-scale 0.5  ############## 8--2 skip connection
        self.trans_block8 = TransitionBlock(32, 32)
        self.dense_block8 = Bottle2neckX(64, 64, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block8s = Bottleneck(64, 64, stride=self.stride, cardinality=self.cardinality)
        self.trans_block8_o = TransitionBlock(128, 32, choice='3')

        ############# Block9-scale 1.0  ############## 9--1 skip connection
        self.trans_block9 = TransitionBlock(32, 32)
        self.dense_block9 = Bottle2neckX(80, 80, self.baseWidth, self.cardinality, self.stride, scale=self.scale)
        #self.dense_block9s = Bottleneck(80, 80, stride=self.stride, cardinality=self.cardinality)
        self.trans_block9_o = TransitionBlock(160, 16, choice='3')

        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.zout = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.refineclean2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, xin):
        x = self.conv_input(xin)
        # Size - 1.0
        x1 = (self.dense_block1(x))

        # Size - 0.5
        x2_i = self.trans_block2(x1)
        x2_i = self.dense_block2(x2_i)
        x2 = self.trans_block2_o(x2_i)

        # Size - 0.25
        x3_i = self.trans_block3(x2)
        x3_i = self.dense_block3(x3_i)
        x3 = self.trans_block3_o(x3_i)

        # Size - 0.125
        x4_i = self.trans_block4(x3)
        x4_i = self.dense_block4(x4_i)
        x4 = self.trans_block4_o(x4_i)

        x5_i = self.dense_block5(x4)
        x5 = self.trans_block5_o(x5_i)

        x6_i = self.dense_block6(x5)
        x6 = self.trans_block6_o(x6_i)
        z = self.zout(self.relu(x6))

        x7_i = self.trans_block7(z)
        # print(x4.size())
        # print(x7_i.size())
        x7_i = self.dense_block7(torch.cat([x7_i, x3], 1))
        x7 = self.trans_block7_o(x7_i)

        x8_i = self.trans_block8(x7)
        x8_i = self.dense_block8(torch.cat([x8_i, x2], 1))
        x8 = self.trans_block8_o(x8_i)

        x9_i = self.trans_block9(x8)
        x9_i = self.dense_block9(torch.cat([x9_i, x1, x], 1))
        x9 = self.trans_block9_o(x9_i)

        x11 = x - self.relu((self.conv_refin(x9)))
        residual = self.tanh(self.refine3(x11))
        clean = residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.sig(self.refineclean2(clean))

        return clean, z
