import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
n = 1000  # 时间点的数量
t = np.linspace(0, 1, n)  # 时间序列，从0到1
dt = 1 / (n - 1)  # 时间步长
N = 400  # 生成的布朗桥的数量

# 高斯分布参数
mu_start = [0, 0, 0]  # 起始点高斯分布的均值
cov_start = np.diag([2, 2, 2])  # 起始点高斯分布的协方差矩阵

# 初始化起始点
x_start = np.random.multivariate_normal(mu_start, cov_start, N)

mu_end_list = [[3, 3, 3], [3, -3, 3], [-3, 3, 3], [-3, -3, 3]]  # 四个小高斯分布的均值
sigma_end = np.diag([1, 1, 1])  # 终点小高斯分布的协方差矩阵

# 初始化终点
x_end = np.zeros((N, 3))
for i in range(4):
    x_end[i * 100:(i + 1) * 100] = np.random.multivariate_normal(mu_end_list[i], sigma_end, 100)

# 定义漂移项函数mu，对于布朗桥，漂移项应为 (end - current) / (1 - t)，排除t=1的情况
def mu(y, end, t):
    if t == 1:
        return np.zeros_like(y)  # 当t等于1时，漂移项为0
    return (end - y) / (1 - t)

# 定义方差函数sigma，对于布朗桥，方差项通常为sqrt(t)
def sigma(t):
    return np.sqrt(t)

# 生成正态分布随机数，考虑布朗运动的增量
dW = np.sqrt(dt) * np.random.randn(n, N, 3)

# 初始化N个布朗桥的状态
X = np.zeros((n, N, 3))
X[0, :] = x_start

# 模拟 SDE
for i in range(1, n):
    for j in range(N):
        X[i, j] = X[i - 1, j] + mu(X[i - 1, j], x_end[j], t[i]) * dt + sigma(t[i]) * dW[i, j]

# 准备训练数据
# 使用所有时间点的所有布朗桥的数据进行训练，排除最后一个时间点t=1
train_X = []
train_y = []
for i in range(1, n - 1):  # 排除最后一个时间点
    for j in range(N):
        train_X.append([X[i - 1, j, 0], X[i - 1, j, 1], X[i - 1, j, 2], t[i]])
        train_y.append(mu(X[i - 1, j], x_end[j], t[i]))

# 优化后的 Tensor 转换
train_X = torch.tensor(np.array(train_X), dtype=torch.float32).to(device)
train_y = torch.tensor(np.array(train_y), dtype=torch.float32).to(device)

# 定义神经网络模型
class DriftNN(nn.Module):
    def __init__(self):
        super(DriftNN, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型和优化器
model = DriftNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练神经网络
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 使用神经网络替代漂移系数进行模拟
X_nn = np.zeros((n, N, 3))
X_nn[0, :] = x_start

for i in range(1, n):
    for j in range(N):
        if t[i] == 1:
            X_nn[i, j] = x_end[j]  # 最后一个时间点直接设置为终点
        else:
            drift = model(torch.tensor([[X_nn[i - 1, j, 0], X_nn[i - 1, j, 1], X_nn[i - 1, j, 2], t[i]]], dtype=torch.float32).to(device)).cpu().detach().numpy()[0]
            X_nn[i, j] = X_nn[i - 1, j] + drift * dt + sigma(t[i]) * dW[i, j]

# 设置支持中文的字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 使用 matplotlib 绘制结果可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for j in range(N):
    ax.plot(X_nn[:, j, 0], X_nn[:, j, 1], X_nn[:, j, 2], alpha=0.5, label='NN' if j == 0 else "")

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('使用神经网络替代漂移系数模拟三维布朗桥')
plt.legend()
plt.grid(True)
plt.show()
