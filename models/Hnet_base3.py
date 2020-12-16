import torch
import torch.nn as nn
from utils.init_weights import init_weights

# based on the Unet
# downsample: two conv + BN+ LeakyRelu, maxpool size/2
# upsample:
# 1216 这个待定！！！！

class HNet(nn.Module):
    def __init__(self, colordim=6):
        super(HNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]  # 特征的通道数

        # 特征提取
        self.downsamp1 = DoubleConv(colordim, filters[0], filters[0])  # 3, 64, 64
        self.downsamp2 = DoubleConv(filters[0], filters[1], filters[1])  # 64, 128, 128
        self.downsamp3 = DoubleConv(filters[1], filters[2], filters[2])  # 128,256,256
        self.downsamp4 = DoubleConv(filters[2], filters[3], filters[3])  # 特征维度256,512,512
        self.downsamp5 = DoubleConv(filters[3], filters[4], filters[3])  # 特征维度512,1024,512

        self.upsamp4 = DoubleConv(filters[4], filters[3], filters[2])  # 特征维度1024,512,256
        self.upsamp3 = DoubleConv(filters[3], filters[2], filters[1])  # 特征维度512, 256,128
        self.upsamp2 = DoubleConv(filters[2], filters[1], filters[0])  # 特征维度256, 128, 64
        self.upsamp1 = DoubleConv(filters[1], filters[0], 32)  # 特征维度128,64,32

        self.max = nn.MaxPool2d(2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(32, 3, 1, stride=1, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.downsamp1(x)
        x2 = self.max(x1)

        x2 = self.downsamp2(x2)
        x3 = self.max(x2)

        x3 = self.downsamp3(x3)
        x4 = self.max(x3)

        x4 = self.downsamp4(x4)
        x5 = self.max(x4)

        x5 = self.downsamp5(x5)

        x4_1 = self.up(x5)
        x4_1 = self.upsamp4(torch.cat((x4_1, x4), 1))

        x3_1 = self.up(x4_1)
        x3_1 = self.upsamp3(torch.cat((x3_1, x3), 1))

        x2_1 = self.up(x3_1)
        x2_1 = self.upsamp2(torch.cat((x2_1, x2), 1))

        x1_1 = self.up(x2_1)
        x1_1 = self.upsamp1(torch.cat((x1_1, x1), 1))

        out = self.conv(x1_1)
        out = self.activation(out)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, inner, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inner, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner),  # 添加了BN层
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inner, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        for m in self.children():  # self.children()存储网络结构的子层模块，一层一层
            init_weights(m, init_type='kaiming')  # 对每一层参数进行初始化

    def forward(self, x):
        out = self.conv(x)
        return out


