# encoding: utf-8
import torch.nn as nn
import torch
# 论文： Simultaneous convolutional neural network for highly efficient image steganography
# 5个卷积层，conv size/2， k = 2, s = 2, p = 0
# 5个反卷积层， deconv size*2 k= 2, s = 2, p = 0

# 为了解决棋盘伪影---使用 sub-pixel convolution 代替反卷积
# class torch.nn.PixleShuffle(upscale_factor)
# 输入: (B,C x upscale_factor ^2 ,H,W)
# 输出: (B,C,H x upscale_factor,W x upscale_factor)


class HNet(nn.Module):
    def __init__(self, colordim=6):
        super(HNet, self).__init__()
        # filters = [50, 100, 200, 400]  # 特征的通道数
        filters = [64, 128, 256, 512]  # 特征的通道数
        # 特征提取
        self.layer1 = nn.Sequential(
            nn.Conv2d(colordim, filters[0], kernel_size=2, stride=2, padding=0),  # size/2
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=2, stride=2, padding=0),  # size/2
            nn.BatchNorm2d(filters[3]),  # 400
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(filters[3], 3072, kernel_size=2, stride=2, padding=0),  # size/2
            # 通道数3072，为了上采样
            nn.ReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )

        # 上采样
        self.layer6 = nn.Sequential(
            nn.PixelShuffle(2),  # 输出的768通道，输入应该是3072 size*2
            nn.BatchNorm2d(768),
        )
        self.layer7 = nn.Sequential(
            nn.PixelShuffle(2),  # 输出的192通道，输入应该是768  size*2
            nn.BatchNorm2d(192),
        )
        self.layer8 = nn.Sequential(
            nn.PixelShuffle(2),  # 输出的48通道，输入应该是192  size*2
            nn.BatchNorm2d(48),
        )
        self.layer9 = nn.Sequential(
            nn.PixelShuffle(2),  # 输出的12通道，输入应该是48  size*2
            nn.BatchNorm2d(12),
        )
        self.layer10 = nn.Sequential(
            nn.PixelShuffle(2),  # 输出的3通道，输入应该是12  size*2
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(x5)
        x6 = self.relu(torch.cat([x6, x4], dim=1))

        x7 = self.layer7(x6)
        x7 = self.relu(torch.cat([x7, x3], dim=1))

        x8 = self.layer8(x7)
        x8 = self.relu(torch.cat([x8, x2], dim=1))

        x9 = self.layer9(x8)
        x9 = self.relu(torch.cat([x9, x1], dim=1))

        x9 = self.layer10(x9)

        return x9