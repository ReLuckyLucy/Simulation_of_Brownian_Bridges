import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 设置字体和避免负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
N = 100  # 模拟路径数量
n = 100  # 每个路径的点数
T = 1    # 总时间
t = np.linspace(0, T, n)  # 时间序列
dt = T / (n - 1)  # 时间步长

# 构造"S"形起点分布
def generate_S_shape(N, noise=0.1, center=(0, 0, 0)):
    t_s = np.linspace(0, 2 * np.pi, N)
    x_s = np.sin(t_s) * 4 + 1 + np.random.normal(0, noise, N)  # x坐标变化
    y_s = t_s - np.pi + np.random.normal(0, noise, N)  # y坐标变化
    z_s = np.zeros_like(t_s) + np.random.normal(0, noise, N)  # z坐标不变
    return np.vstack((x_s, y_s, z_s)).T

# 设置S形的中心在左边
center_S = (-10, 0, 0)
x0 = generate_S_shape(N, center=center_S)

# 构造两个半月形的终点分布
def generate_moon(N, noise=0.1, radius=5, center_up=(0, 0, 0), center_down=(0, 0, 0)):
    t_s = np.linspace(0, np.pi, N // 2)
    x_s1 = radius * np.cos(t_s) + center_up[0] + np.random.normal(0, noise, N // 2)
    y_s1 = radius * np.sin(t_s) + center_up[1] + np.random.normal(0, noise, N // 2)
    z_s1 = np.random.normal(center_up[2], noise, N // 2)
    x_s2 = radius * np.cos(t_s + np.pi) + center_down[0] + np.random.normal(0, noise, N // 2)
    y_s2 = radius * np.sin(t_s + np.pi) + center_down[1] + np.random.normal(0, noise, N // 2)
    z_s2 = np.random.normal(center_down[2], noise, N // 2)
    return np.vstack((x_s1, y_s1, z_s1)).T, np.vstack((x_s2, y_s2, z_s2)).T

# 设置上半个月亮的中心在负半轴，设置下半个月亮的中心在正半轴
center_up = (0, 5, 0)
center_down = (0, -5, 0)
xf_up, xf_down = generate_moon(N, radius=5, center_up=center_up, center_down=center_down)
xf = np.vstack((xf_up, xf_down))

# 转换为PyTorch张量并移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x0 = torch.tensor(x0, dtype=torch.float32).to(device)
xf = torch.tensor(xf, dtype=torch.float32).to(device)

# 布朗桥模拟函数
def simulate_brownian_bridge(x0, xf, n, T, device):
    dt = T / (n - 1)
    x = torch.zeros((N, n, 3), device=device)
    x[:, 0, :] = x0
    drift = (xf - x0) / T
    sigma = np.sqrt(dt)
    for i in range(1, n):
        dW = torch.normal(0, sigma, size=(N, 3), device=device)
        x[:, i, :] = x[:, i-1, :] + drift * dt + dW
    return x

# 模拟布朗桥
x_bb = simulate_brownian_bridge(x0, xf, n, T, device)

# 将结果从GPU移到CPU以进行可视化
x_bb = x_bb.cpu().numpy()
x0 = x0.cpu().numpy()
xf = xf.cpu().numpy()

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始点和终止点
ax.scatter(np.zeros(N), x0[:, 0], x0[:, 1], color='red', label='初始点 (t=0)', alpha=0.6)
ax.scatter(np.ones(N) * T, xf[:, 0], xf[:, 1], color='black', label='终止点 (t=T)', s=10, alpha=0.6)

# 画布朗桥路径
for i in range(N):
    ax.plot(t, x_bb[i, :, 0], x_bb[i, :, 1], alpha=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$x_2$')
ax.legend()
ax.set_title('布朗桥模拟路径与“S”形起点和两个半月形终点')

plt.show()
