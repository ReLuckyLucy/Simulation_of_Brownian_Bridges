import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.output(dec1)

    def pool(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)


# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5
P = 784  # MNIST图片尺寸为28x28, 因此P平方维为784

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

# 下载并加载MNIST数据集，只保留数字0的数据
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
zero_data = [data for data in train_dataset if data[1] == 0]
train_loader = DataLoader(dataset=zero_data, batch_size=batch_size, shuffle=True)

# 定义布朗桥采样函数
def brownian_bridge(start, end, num_steps):
    bridge = []
    for t in range(num_steps):
        alpha = t / (num_steps - 1)
        sample = (1 - alpha) * start + alpha * end + np.sqrt(alpha * (1 - alpha)) * np.random.normal(size=start.shape)
        bridge.append(sample)
    return np.array(bridge)


# 初始化U-Net模型
model = UNet(in_channels=1, out_channels=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练U-Net模型
for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.view(-1, 1, 28, 28)
        batch_size = images.size(0)

        # 从高斯分布中抽取样本
        gaussian_samples = torch.randn(batch_size, 1, 28, 28)

        # 生成布朗桥样本
        num_steps = 10
        brownian_samples = []
        for i in range(batch_size):
            bridge = brownian_bridge(gaussian_samples[i].numpy().flatten(), images[i].numpy().flatten(), num_steps)
            brownian_samples.append(torch.tensor(bridge, dtype=torch.float32).view(num_steps, 1, 28, 28))
        brownian_samples = torch.stack(brownian_samples)

        # 训练模型
        for t in range(num_steps):
            outputs = model(brownian_samples[:, t])
            loss = criterion(outputs, brownian_samples[:, (t + 1) % num_steps])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'unet_mnist_zero.pth')
