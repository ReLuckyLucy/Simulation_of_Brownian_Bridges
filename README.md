#  Simulation_of_Brownian_Bridges

<div align="center">
 <img alt="ollama" height="200px" src="https://github.com/ReLuckyLucy/Simulation-of-Brownian-Bridges/blob/main/img/suiji.png">
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
