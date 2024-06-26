import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体和避免负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
N = 400  # 总的模拟路径数量
n = 100  # 每个路径的点数
T = 1    # 总时间
t = np.linspace(0, T, n)  # 时间序列
dt = T / (n - 1)  # 时间步长

# 构造初始时刻t=0的正态分布
mean_0 = [0, 0, 0]
cov_0 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
x0 = multivariate_normal(mean_0, cov_0).rvs(size=N)

# 在大T时刻构造四个小的正态分布并抽取终点
means_T = [[4, 4, 4], [-4, 4, 4], [4, -4, -4], [-4, -4, -4]]
cov_T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 小的正态分布的协方差矩阵
xf = np.concatenate([multivariate_normal(mean, cov_T).rvs(size=N//4) for mean in means_T])

# 布朗桥模拟函数
def simulate_brownian_bridge(x0, xf, n, T):
    dt = T / (n - 1)
    x = np.zeros((N, n, 3))
    x[:, 0, :] = x0
    drift = (xf - x0) / T
    sigma = np.sqrt(dt)
    for i in range(1, n):
        dW = np.random.normal(0, sigma, size=(N, 3))
        x[:, i, :] = x[:, i-1, :] + drift * dt + dW
    return x

# 模拟布朗桥
x_bb = simulate_brownian_bridge(x0, xf, n, T)

# 准备训练数据
X_train = np.hstack((np.tile(t[:-1], (N, 1)).reshape(-1, 1), x_bb[:, :-1].reshape(-1, 3)))
drift = (xf - x0) / T
y_train = np.tile(drift, (n-1, 1)).reshape(-1, 3) * dt

# 转换为PyTorch张量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

# 定义神经网络模型
class DriftNN(nn.Module):
    def __init__(self):
        super(DriftNN, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = DriftNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train[:, 1:])
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 使用神经网络预测漂移系数并模拟新路径
x_nn = np.zeros_like(x_bb)
x_nn[:, 0, :] = x0
sigma = np.sqrt(dt)  # 重新定义sigma
for i in range(1, n):
    input_features = torch.tensor(np.hstack((np.full((N, 1), t[i]), x_nn[:, i-1, :])), dtype=torch.float32).to(device)
    with torch.no_grad():
        drift_nn = model(input_features).cpu().numpy()
    dW = np.random.normal(0, sigma, size=(N, 3))
    x_nn[:, i, :] = x_nn[:, i-1, :] + np.hstack((np.zeros((N, 1)), drift_nn)) * dt + dW

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始点和终止点
ax.scatter(np.zeros(N), x0[:, 0], x0[:, 1], color='blue', label='初始点 (t=0)')
ax.scatter(np.ones(N) * T, xf[:, 0], xf[:, 1], color='red', label='终止点 (t=T)')

# 画布朗桥路径
for i in range(N):
    ax.plot(t, x_bb[i, :, 0], x_bb[i, :, 1], alpha=1)

# 画神经网络模拟路径
# for i in range(N):
#     ax.plot(t, x_nn[i, :, 0], x_nn[i, :, 1], color='green', alpha=0.1, linestyle='--')

ax.set_xlabel('时间 t')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$x_2$')
ax.legend()
ax.set_title('布朗桥模拟路径与神经网络模拟路径')

plt.show()
