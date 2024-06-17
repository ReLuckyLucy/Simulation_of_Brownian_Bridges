import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数设置
x0 = 1  # 初始值
n = 1000  # 时间点的数量
t = np.linspace(0, 1, n)  # 时间序列，从0到1
dt = 1 / (n - 1)  # 时间步长
N = 10  # 生成的布朗桥的数量
a = 10  # 目标结束值

# 初始化
x = np.ones((N, n)) * x0

# 定义 mu 和 sigma 函数
def mu(y, t, a):
    return (a - y) / (1 - t)

def sigma(t):
    return 1

# 生成正态分布随机数
dW = np.sqrt(dt) * np.random.randn(N, n)

# 模拟 SDE
for i in range(n - 1):
    x[:, i + 1] = x[:, i] + mu(x[:, i], t[i], a) * dt + sigma(t[i]) * dW[:, i]

# 布朗桥在t=1时的值应为a，手动设置
x[:, -1] = a

# 使用 matplotlib 绘制结果可视化
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, x[i], alpha=1)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(u'Simulation of Brownian Bridges布朗桥')
plt.grid(True)
plt.show()