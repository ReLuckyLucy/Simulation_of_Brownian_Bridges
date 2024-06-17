import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib以正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
n = 1000  # 时间点的数量
t = np.linspace(0, 1, n)  # 时间序列，从0到1
dt = 1 / (n - 1)  # 时间步长
N = 200  # 生成的布朗桥的数量

# 高斯分布参数
mu_start = 0  # 起始点高斯分布的均值
sigma_start = 1  # 起始点高斯分布的方差

# 初始化起始点
x_start = np.random.normal(mu_start, sigma_start, N)



mu_end_pos = 3  # 终点前一半个点高斯分布的均值
sigma_end_pos = 1  # 终点前一半个点高斯分布的方差

mu_end_neg = -3  # 终点后一半个点高斯分布的均值
sigma_end_neg = 1  # 终点后一半个点高斯分布的方差

# 初始化终点
x_end_pos = np.random.normal(mu_end_pos, sigma_end_pos, N // 2)
x_end_neg = np.random.normal(mu_end_neg, sigma_end_neg, N // 2)

# 为前一半布朗桥分配正的终点，后一半分配负的终点
x_end = np.concatenate((x_end_pos, x_end_neg))

# 定义漂移项函数mu，对于布朗桥，漂移项应为 (end - current) / (1 - t)
def mu(y, end, t):
    return (end - y) / (1 - t)

# 定义方差函数sigma，对于布朗桥，方差项通常为sqrt(t)
def sigma(t):
    return np.sqrt(t)

# 生成正态分布随机数，考虑布朗运动的增量
dW = np.sqrt(dt) * np.random.randn(n, N)


# 初始化N个布朗桥的状态
X = np.zeros((n, N))
X[0, :] = x_start

# 模拟 SDE
for i in range(1, n):
    for j in range(N):
        end = x_end[j]  # 获取对应布朗桥的终点
        X[i, j] = X[i - 1, j] + mu(X[i - 1, j], end, t[i]) * dt + sigma(t[i]) * dW[i, j]

# 使用 matplotlib 绘制结果可视化
plt.figure(figsize=(10, 6))
for j in range(N):
    plt.plot(t, X[:, j], alpha=1)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(u'Simulation of Brownian Bridges 高斯分布布朗桥')
plt.grid(True)
plt.show()