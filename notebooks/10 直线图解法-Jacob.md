---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 直线图解法 — Jacob

以 $s\sim \lg t$ 为例：

$$
s\sim\frac{0.183Q}{T} \lg t+\frac{0.183Q}{T}\lg\frac{2.25T}{r^2S}
$$

写成点斜式

$$
s\sim i\lg t - i\lg t_0
$$

通过调整 $t_0$ 与 $i$ 拟合直线，并用如下公式计算参数：

$$
T=\frac{0.183Q}{i},\quad  S=\frac{2.25Tt_0}{r^2}
$$

**程序设计**

程序设计流程为 “库导入 $\implies$ 数据准备 -> 绘图准备 $\implies$ `widgets.interact` 互动”。

```python
%matplotlib widget

import numpy as np
import math as math
import matplotlib.pyplot as plt
import ipywidgets as widgets

# 控制小数的显示精度
np.set_printoptions(precision=4)

# 准备数据
Q = 528/1440  # m^3/min
r = 90        # m

t = np.array([1, 2, 4, 6, 9, 20, 30, 40, 50, 60,
              90, 120, 150, 360, 550, 720])     # min
s = np.array([2.5, 3.9, 6.1, 8.0, 10.6, 16.8, 20.0, 22.6, 24.7, 26.4,
              30.4, 33.0, 35.0, 42.6, 44.0, 44.5])/100   # m

#绘图界限
ymin = 0.0
ymax = math.ceil(max(s*10))/10
imin = math.floor(math.log10(min(t)))
imax = math.ceil(math.log10(max(t)))  
xmin = 10**imin
xmax =  10**imax

# 最小二乘法求初始参数
A = np.vstack([np.ones(len(t)), np.log10(t)]).T  # 形成系数
# 求解方程组
beta = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, s))
i0 = round(beta[1],3) # 斜率   
t0 = round(np.float_power(10, -beta[0]/beta[1]),1) #截距

T = 0.183*Q/i0
S = 2.25*T*t0/r**2

# Plot the data
def line_fit(a, b):  # a - 截距，b - 斜率
    global i0, t0, T, S    # 全局变量
    # 设置图形
    plt.style.use('default')
    fig, ax = plt.subplots(dpi=100)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    plt.xlabel('$\log t$')
    plt.ylabel('$s$')
    ax.grid(True)
    # 绘观测数据散点图
    ax.plot(t, s, '*', label="观测值")
    # 绘直线
    x=np.linspace(xmin, xmax,100)
    ax.plot(x, b*np.log10(x/a), label="拟合直线")
    # 计算调整后的参数
    T = 0.183*Q/b
    S = 2.25*T*a/r**2    
    
    plt.legend(
        prop={'family': 'Simsun'}, handlelength=6,
        loc=4,title="图例",
        title_fontproperties={'family': 'KaiTi'})
    # 指定图标题，显示中文
    ax.set_title("直线图解法", fontproperties={'family': 'KaiTi'})   

    plt.show()
    # 输出参数
    print('  T = {:.4f} m^2/min'.format(T))
    print('  S = {:.4e}'.format(S))

# 小部件互动
widgets.interact(
    line_fit,
    a = widgets.FloatSlider(
        value=t0, min=1, max=t0*5.0, step=.1,
        description='$t_0$ [-]:',
        continuous_update=False,
        readout_format='.1f',
        disabled=False),
    b = widgets.FloatSlider(
        value=i0, min=i0-0.1, max=i0+0.1, step=0.001,
        #value=i0, min=0.5*i0, max=2*i0, step=0.001,
        description='$slope$ [-]:',
        continuous_update=False,
        readout_format='.3f',
        disabled=False)
    );
```

```python

```
