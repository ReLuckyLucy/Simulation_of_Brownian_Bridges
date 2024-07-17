import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

# 构造初始时刻t=0的正态分布
mean_0 = [0, 0, 0]
cov_0 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
x0 = multivariate_normal(mean_0, cov_0).rvs(size=N)

# 在大T时刻构造两个半月形的终点分布
def generate_moon(N, noise=0.1, radius=5, center_up=(0,0,0), center_down=(0,0,0)):
    t_s = np.linspace(0, np.pi, N // 2)
    x_s1 = radius * np.cos(t_s) + center_up[0] + np.random.normal(0, noise, N // 2)
    y_s1 = radius * np.sin(t_s) + center_up[1] + np.random.normal(0, noise, N // 2)
    z_s1 = np.random.normal(center_up[2], noise, N // 2)
    x_s2 = radius * np.cos(t_s + np.pi) + center_down[0] + np.random.normal(0, noise, N // 2)
    y_s2 = radius * np.sin(t_s + np.pi) + center_down[1] + np.random.normal(0, noise, N // 2)
    z_s2 = np.random.normal(center_down[2], noise, N // 2)
    return np.vstack((np.vstack((x_s1, y_s1, z_s1)).T, np.vstack((x_s2, y_s2, z_s2)).T))

# 设置上半个月亮的中心在负半轴，设置下半个月亮的中心在正半轴
center_up = (-3, 0, 0)
center_down = (3, 0, 0)
xf = generate_moon(N, radius=5, center_up=center_up, center_down=center_down)

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
X = np.hstack((np.tile(t[:-1], (N, 1)).reshape(-1, 1), x_bb[:, :-1].reshape(-1, 3)))
drift = (xf - x0) / T
y = np.tile(drift, (n-1, 1)).reshape(-1, 3) * dt

# 训练神经网络，逐维训练
mlp_x0 = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42)
mlp_x1 = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42)
mlp_x2 = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', random_state=42)

# 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

mlp_x0.fit(X_train, y_train[:, 0])
mlp_x1.fit(X_train, y_train[:, 1])
mlp_x2.fit(X_train, y_train[:, 2])

# 使用神经网络预测漂移系数并模拟新路径
x_nn = np.zeros_like(x_bb)
x_nn[:, 0, :] = x0
sigma = np.sqrt(dt)  # 重新定义sigma
for i in range(1, n):
    input_features = np.hstack((np.full((N, 1), t[i]), x_nn[:, i-1, :]))
    drift_x0 = mlp_x0.predict(input_features)
    drift_x1 = mlp_x1.predict(input_features)
    drift_x2 = mlp_x2.predict(input_features)
    drift_nn = np.vstack((drift_x0, drift_x1, drift_x2)).T
    dW = np.random.normal(0, sigma, size=(N, 3))
    x_nn[:, i, :] = x_nn[:, i-1, :] + drift_nn * dt + dW

# 可视化结果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始点和终止点
ax.scatter(np.zeros(N), x0[:, 0], x0[:, 1], color='blue', label='初始点 (t=0)')
ax.scatter(np.ones(N) * T, xf[:, 0], xf[:, 1], color='red', label='终止点 (t=T)', s=10)  # 调整点的大小以便更易分辨

# 画布朗桥路径
for i in range(N):
    ax.plot(t, x_bb[i, :, 0], x_bb[i, :, 1], alpha=1)

# 画神经网络模拟路径
for i in range(N):
    ax.plot(t, x_nn[i, :, 0], x_nn[i, :, 1], color='green', alpha=0.5, linestyle='--')

ax.set_xlabel('时间 t')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$x_2$')
ax.legend()
ax.set_title('布朗桥模拟路径与两个半月形终点')

plt.show()
