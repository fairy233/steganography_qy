# encoding: utf-8
import torch.nn as nn
import torch
# 论文： Simultaneous convolutional neural network for highly efficient image steganography
# 5个卷积层，conv size/2， k = 2, s = 2, p = 0
# 5个反卷积层， deconv size*2 k= 2, s = 2, p = 0

# 改变卷积核的参数！！   使用 k = 4, s = 2, p = 1


class HNet(nn.Module):
    def __init__(self, colordim=6):
        super(HNet, self).__init__()
        filters = [64, 128, 256, 512]  # 特征的通道数

        # 特征提取
        self.layer1 = nn.Sequential(
            nn.Conv2d(colordim, filters[0], kernel_size=4, stride=2, padding=1),  # size/2=128  3 64
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=4, stride=2, padding=1),  # size/2=64 64 128
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=4, stride=2, padding=1),  # size/2=32 128 256
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=4, stride=2, padding=1),  # size/2=16 256 512
            nn.BatchNorm2d(filters[3]),  # 400
            nn.LeakyReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1),  # size/2=8 512 512
            nn.ReLU(inplace=True),  # inplace-选择是否进行覆盖运算
        )

        # 上采样
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1),   # 16 16 512
            nn.BatchNorm2d(filters[3]),
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(filters[3] * 2, filters[2], kernel_size=4, stride=2, padding=1),  # 32 32 1024 -> 256
            nn.BatchNorm2d(filters[2]),
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(filters[2] * 2, filters[1], kernel_size=4, stride=2, padding=1),  # 64 64 512 -> 128
            nn.BatchNorm2d(filters[1]),
        )
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(filters[1] * 2, filters[0], kernel_size=4, stride=2, padding=1),  # 128 128 256 -> 64
            nn.BatchNorm2d(filters[0]),
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(filters[0] * 2, 3, kernel_size=4, stride=2, padding=1),  # 256 256 64 -> 3
            nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x1 = self.layer1(x)  # 128 128 64
        x2 = self.layer2(x1)  # 64 64 128
        x3 = self.layer3(x2)  # 32 32 256
        x4 = self.layer4(x3)  # 16 16 512
        x5 = self.layer5(x4)  # 8 8 512

        x6 = self.layer6(x5)  # 16 16 512
        x6 = self.relu(torch.cat([x6, x4], dim=1))  # 16 16 1024

        x7 = self.layer7(x6)  # 32 32 256
        x7 = self.relu(torch.cat([x7, x3], dim=1))  # 32 32 512

        x8 = self.layer8(x7)  # 64 64 128
        x8 = self.relu(torch.cat([x8, x2], dim=1))  # 64 64 256

        x9 = self.layer9(x8)  # 128 128 64
        x9 = self.relu(torch.cat([x9, x1], dim=1))  # 128 128 128

        x9 = self.layer10(x9)  # 256 256 3

        return x9