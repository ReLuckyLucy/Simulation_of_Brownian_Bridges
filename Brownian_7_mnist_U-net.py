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
        # 定义编码器（Encoder）部分
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),    # 输入通道数为in_channels，输出通道数为56
            DoubleConv(64, 128),            # 输入通道数为56，输出通道数为112
            DoubleConv(128, 256),           # 输入通道数为112，输出通道数为224
            DoubleConv(256, 512)            # 输入通道数为224，输出通道数为448
        ])
        
        # 定义解码器（Decoder）部分
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 转置卷积，将448通道映射为224通道
            DoubleConv(512, 256),           # 输入通道数为448（来自上采样+跳跃连接），输出通道数为224
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 转置卷积，将224通道映射为112通道
            DoubleConv(256, 128),           # 输入通道数为224（来自上采样+跳跃连接），输出通道数为112
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 转置卷积，将112通道映射为56通道
            DoubleConv(128, 64),            # 输入通道数为112（来自上采样+跳跃连接），输出通道数为56
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)  # 转置卷积，将56通道映射为输出通道数
        ])

    def forward(self, x):
        # Encoder过程
        features = []
        for down in self.encoder:
            x = down(x)                # 应用双重卷积块
            features.append(x)         # 将结果保存到features列表中，以便后续使用
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)  # 最大池化，缩小特征图大小

        # Decoder过程
        for i, up in enumerate(self.decoder):
            if i % 2 == 0:
                x = up(x)              # 使用转置卷积进行上采样
                skip_features = features[-(i // 2 + 1)]  # 获取对应的编码器特征图
                # 对skip_features进行调整以匹配x的大小
                skip_features = self.adjust_skip_features(skip_features, x)
                x = torch.cat([skip_features, x], dim=1)  # 进行跳跃连接
            else:
                x = up(x)               # 应用双重卷积块

        return x

    def adjust_skip_features(self, skip_features, x):
        """
        调整skip_features的大小以匹配x的大小。
        """
        _, _, height, width = x.size()
        return nn.functional.interpolate(skip_features, size=(height, width), mode='bilinear', align_corners=True)

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),                  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))    # 归一化处理
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 初始化UNet模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否可用GPU加速
model = UNet(in_channels=1, out_channels=1).to(device)   # 实例化UNet模型并移动到GPU或CPU
criterion = nn.MSELoss()    # 损失函数为均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001)   # Adam优化器，学习率为0.001

# 训练UNet模型
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)   # 将输入数据移动到GPU或CPU
        
        # 生成高斯噪声
        noise = torch.randn_like(inputs) * 0.1
        inputs += noise    # 将高斯噪声添加到输入数据中
        
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播、反向传播、优化
        outputs = model(inputs)
        loss = criterion(outputs, inputs)   # 计算损失
        loss.backward()    # 反向传播
        optimizer.step()   # 优化器更新模型参数

        # 输出统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量数据打印一次损失
            print(f'Epoch [{epoch + 1}, {i + 1}], Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 生成新的高斯噪声样本并可视化生成的图像
model.eval()    # 设置模型为评估模式
with torch.no_grad():
    sample_noise = torch.randn(1, 1, 28, 28).to(device) * 0.1   # 生成高斯噪声
    generated_image = model(sample_noise)    # 使用模型生成图像

# 可视化生成的图像
plt.imshow(generated_image[0][0].cpu().numpy(), cmap='gray')  # 将张量转换为NumPy数组并绘制灰度图像
plt.title('Generated Image')   # 设置图像标题
plt.axis('off')    # 关闭坐标轴
plt.show()         # 显示图像
