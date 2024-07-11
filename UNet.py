import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)

        y4 = self.decoder[0](x5)
        y3 = self.decoder[1](torch.cat([y4, x4], 1))
        y2 = self.decoder[2](torch.cat([y3, x3], 1))
        y1 = self.decoder[3](torch.cat([y2, x2], 1))
        y0 = self.decoder[4](torch.cat([y1, x1], 1))

        return self.decoder[5](y0)

# 创建一个UNet模型实例
model = UNet(in_channels=1, out_channels=1)
print(model)
