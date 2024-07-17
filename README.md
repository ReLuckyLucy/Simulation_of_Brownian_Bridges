#  Simulation_of_Brownian_Bridges

<div align="center">
 <img alt="ollama" height="200px" src="https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/logo.png">
</div>
基于布朗运动模型模拟的布朗桥方程式及通过神经网络构建模型进行扩散模型的布朗模拟

##  1.基于SDE方程式构建的布朗运动模拟

![模拟布朗运动](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/模拟布朗运动.png)

我们通过SDE方程式，实现随机变量

![pseudo_code](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/pseudo_code.jpg)

## 2.基于SDE方程式构建的布朗桥模拟

布朗桥定义如下

![Brownian_motion基本定义,](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/Brownian_motion.jpg)

我们基于此，在代码中进行调整

![模拟布朗桥](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/模拟布朗桥.png)

## 3.基于高斯分布的布朗桥模拟

我们把200个起始点从均值为零，方差为一的高斯分布中抽取，
我们把200个终点前100个点从均值为三，方差为1的高斯分布中抽取，后100个点从均值为-3，方差为1的高斯分布中抽取，
然后用布朗桥的代码把每一个配好对的两个点用bridge simulation连起来

![高斯分布模拟布朗桥](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/高斯分布模拟布朗桥.png)

## 4.基于神经网络模拟布朗桥

我们继续模拟这个轨迹，但是我们选择不用布朗桥的漂移系数。而是训练一个神经网络，这里我们训练了一个多层感知机模拟这个轨迹

我们定义了一个目标函数，神经网络的参数是\phi，NN的输入是当前时刻t, x_t, 目的是拟合布朗桥的漂移系数，

![使用神经网络替代漂移系数模拟布朗桥](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/使用神经网络替代漂移系数模拟布朗桥.png)

## 5. 训练神经网络生成二维数据分布
> 我们希望通过训练神经网络来生成二维数据分布，具体步骤如下：

###  初始时刻 (t=0):
构造一个二维正态分布，其均值为(0, 0)，协方差矩阵为对角线元素为2，非对角线元素为0的矩阵。

###  目标时刻 (T时刻):
构造四个较小的二维正态分布。
从初始正态分布中抽取400个起始点。
从每个小的正态分布中各抽取100个终点。

### 布朗运动配对:
使用布朗运动将每个起始点和相应的终点配对连接起来。

### 训练神经网络:
训练一个神经网络（NN）来学习对应的漂移函数（Drift Function）。
对于维度大于一的情况，可以逐维度地训练神经网络。

### 重新仿真:
使用训练好的神经网络进行重新仿真，生成新的数据分布。

![3d下布朗桥模型与神经网络模拟路径对比](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/3d下布朗桥模型与神经网络模拟路径对比.png)

## 6. 训练神经网络生成二维数据分布_更改起始点与终点分布
基于任务5的代码，我们进行修改，改变其起始点与终点的分布方式
### 6_1布朗桥模拟路径与同心圆形终点
![3d下布朗桥模型与神经网络模拟路径对比](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/6_1布朗桥模拟路径与同心圆形终点.png)

### 6_2布朗桥模拟路径与“S”形终点
![3d下布朗桥模型与神经网络模拟路径对比](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/6_2布朗桥模拟路径与“S”形终点.png)

### 6_3布朗桥模拟路径与两个半月形终点
![3d下布朗桥模型与神经网络模拟路径对比](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/6_3布朗桥模拟路径与两个半月形终点.png)

### 6_4布朗桥模拟路径与“S”形起点和两个半月形终点
![3d下布朗桥模型与神经网络模拟路径对比](https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/6_4布朗桥模拟路径与“S”形起点和两个半月形终点.png)