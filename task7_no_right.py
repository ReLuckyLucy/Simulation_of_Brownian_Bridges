import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 生成布朗桥
def brownian_bridge(x0, xT, T, N):
    t = np.linspace(0, T, N)
    delta_t = T / (N-1)
    W = np.random.normal(0, np.sqrt(delta_t), (N-1, x0.shape[0]))
    W = np.vstack((np.zeros(x0.shape), np.cumsum(W, axis=0)))
    B_t = x0 + (xT - x0) * t[:, None] / T + W - t[:, None] / T * W[-1]
    return B_t

# 生成数据和模型
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# 定义一些常量
p = 28  # MNIST图像是28x28像素
T = 1.0
N = 1000
input_dim = p**2
output_dim = p**2

# 准备MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 只提取数字"0"的样本
mnist_zero = [(img, label) for img, label in mnist if label == 0]
train_loader = torch.utils.data.DataLoader(mnist_zero, batch_size=64, shuffle=True)

# 初始化生成器
generator = Generator(input_dim, output_dim).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# 训练生成器
epochs = 5
for epoch in range(epochs):
    for imgs, _ in train_loader:
        imgs = imgs.view(imgs.size(0), -1).to('cuda')  # 扁平化图像
        optimizer.zero_grad()

        # 从高斯分布中采样并生成图像
        z = torch.randn(imgs.size(0), input_dim).to('cuda')
        generated_imgs = generator(z)

        loss = criterion(generated_imgs, imgs)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 生成新图像并绘制图例
z = torch.randn(1, input_dim).to('cuda')
generated_img = generator(z).view(p, p).cpu().detach().numpy()

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Gaussian Sample')
plt.imshow(z.view(p, p).cpu().numpy(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Brownian Bridge')
z_flat = z.view(-1).cpu().numpy()
generated_img_flat = generated_img.flatten()
bridge = brownian_bridge(z_flat, generated_img_flat, T, N).reshape((N, p, p))
plt.imshow(bridge[-1], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Generated Image')
plt.imshow(generated_img, cmap='gray')
plt.show()
