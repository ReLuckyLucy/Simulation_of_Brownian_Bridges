import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt  # 添加了 matplotlib 库用于绘图


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 参数设置
x0 = 1
n = 1000
t = np.linspace(0, 1, n)
dt = 1 / (n - 1)
N = 10

# 初始化
x = np.ones((N, n)) * x0

# 定义 mu 和 sigma 函数
def mu(x):
    return np.sin(x)

def sigma(x):
    return 1

# 生成正态分布随机数
dW = np.sqrt(dt) * np.random.randn(N, n)

# 模拟 SDE
for i in range(n - 1):
    x[:, i+1] = x[:, i] + mu(x[:, i]) * dt + sigma(x[:, i]) * dW[:, i]

# 打印结果
print(x)

# 使用 seaborn 绘制结果可视化
#sns.set_palette("pastel", 8)

# 绘制所有路径的折线图
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(t, x[i], alpha=1)  # 使用 alpha 参数使得多条路径更易观察
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Simulation of SDE')
plt.show()