import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义双重卷积块
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

# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 28),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(28, 56),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(56, 112),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(112, 224),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(224, 112, kernel_size=2, stride=2),
            DoubleConv(224, 112),
            nn.ConvTranspose2d(112, 56, kernel_size=2, stride=2),
            DoubleConv(112, 56),
            nn.ConvTranspose2d(56, 28, kernel_size=2, stride=2),
            DoubleConv(56, 28),
            nn.ConvTranspose2d(28, out_channels, kernel_size=2, stride=2)
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

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 初始化UNet模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练UNet模型
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        
        # 生成高斯噪声
        noise = torch.randn_like(inputs) * 0.1
        inputs += noise
        
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播、反向传播、优化
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # 输出统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量数据打印一次损失
            print(f'Epoch [{epoch + 1}, {i + 1}], Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 生成新的高斯噪声样本并可视化生成的图像
model.eval()
with torch.no_grad():
    sample_noise = torch.randn(1, 1, 28, 28).to(device) * 0.1
    generated_image = model(sample_noise)

# 可视化生成的图像
plt.imshow(generated_image[0][0].cpu().numpy(), cmap='gray')
plt.title('Generated Image')
plt.axis('off')
plt.show()
