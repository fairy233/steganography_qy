import torch
import torch.nn as nn
# 论文： A Novel Image Steganography Method via Deep Convolutional Generative Adversarial Networks
# 的判别器。 组成：


class Discriminitor(nn.Module):
    def __init__(self):
        super(Discriminitor, self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=(5,5),
            stride=2,
            padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(5, 5),
            stride=2,
            padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(5, 5),
            stride=2,
            padding=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(5, 5),
            stride=2,
            padding=2
        )
        self.fc=nn.Linear(16384,1,True)

        self.lrelu=nn.LeakyReLU()
        self.sigmoid=nn.Sigmoid()
        self.bn1=nn.BatchNorm2d(256)
        self.bn2=nn.BatchNorm2d(512)
        self.bn3=nn.BatchNorm2d(1024)

    def forward(self,x):
        x=self.conv1(x)
        x=self.lrelu(x)

        x=self.conv2(x)
        #x=self.bn1(x)
        x=self.lrelu(x)

        x=self.conv3(x)
        #x=self.bn2(x)
        x=self.lrelu(x)

        x=self.conv4(x)
        #x=self.bn3(x)
        x=self.lrelu(x)

        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        #x=self.sigmoid(x)

        return x


test=torch.ones(1,3,64,64)
g=Discriminitor()
r=g(test)
print(r.shape)