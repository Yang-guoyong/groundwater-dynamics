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

# 配线法 — Theis 标准曲线

以  $s\sim\frac{t}{r^2}$ 配线法为例。

对 $s=\frac{Q}{4\pi T}W(u)$ 及 $u=\frac{r^2S}{4Tt}$ 取对数，可得到如下的形式

$$
\lg s + \lg\frac{4\pi T}{Q}= \lg W(u),\quad\lg \frac{t}{r^2}+\lg\frac{4T}{S}=\lg\frac{1}{u}
$$

在标准曲线坐标系中移动 $s\sim\frac{t}{r^2}$ 散点图匹配标准曲线，根据平移量计算 $(T,S)$ ：

记坐标平移量为

$$
\Delta \lg s=\lg\frac{4\pi T}{Q},\quad \Delta \lg\frac{t}{r^2}=\lg\frac{4T}{S}
$$

参数计算公式：

$$
T=\frac{Q}{4\pi}10^{\Delta\lg s},\quad S=4T10^{-\Delta\lg\frac{t}{r^2}}
$$

程序中用到了 `numpy`,  `math` , `matplotlib`,  `ipywidgets` 等程序库，相关介绍见联机文档。

程序设计流程为 “库导入 $\implies$ 数据准备 $\implies$ 绘图准备 $\implies$ `widgets.interact` 互动”。



**例**

承压含水层定流量非稳定流群孔抽水试验，抽水孔定流量为 $60m^3/h$，观 1、观 2、 观 3 与观 4 孔距抽水孔分别 $43m$、$140m$、$510m$、$780m$，各观测孔水位降深如下表所示。

| 累计抽水时间(min) | 观 1 降深(m) | 观 2 降深(m) | 观 3 降深(m) | 观 4 降深(m) |
| :----: | :----: | :----: | :----: | :----: |
|10 | 0.73 | 0.16 | 0.04 | 
| 20 | 1.28 | 0.48 |  |  |
| 30 | 1.53 | 0.54 |  |  |
| 40 | 1.72 | 0.65 | 0.06 | | 
| 60 | 1.96 | 0.75 | 0.20 |  |
| 80 | 2.14 | 1.00 | 0.20 | 0.04 |
| 100 | 2.28 | 1.12 | 0.20 |  |
| 120 | 2.39 | 1.22 | 0.21 | 0.08 |
| 150 | 2.54 | 1.36 | 0.24 | 0.09 |
| 210 | 2.77 | 1.55 | 0.40 | 0.16 |
| 270 | 2.99 | 1.70 | 0.53 | 0.25 |
| 330 | 3.10 | 1.83 | 0.63 | 0.34 |
| 400 | 3.20 | 1.89 | 0.65 | 0.42 |
| 450 | 3.26 | 1.98 | 0.73 | 0.50 | 
| 645 | 3.47 | 2.17 | 0.93 | 0.71 |
| 870 | 3.68 | 2.38 | 1.14 | 0.87 |
| 990 | 3.77 | 2.46 | 1.24 | 0.96 |
| 1185 | 3.85 | 2.54 | 1.35 | 1.06  |

试用观 2 孔数据按 $s\sim \lg\frac{t}{r^2}$ 配线求含水层水文地质参数。

```python
%matplotlib widget

# 导入库
import math as math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import exp1

# 导入交互绘图模块
import ipywidgets as widgets

# 控制小数的显示精度
np.set_printoptions(precision=4)

plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

font_legend = {
    "family": "Simsun",
    "weight": "normal",
    "size": 20,
    "style": "italic",  # 使字变斜
}

# 准备数据, 根据配线法的种类，需要专门准备数据
Q = 60 / 60  # m^3/min
r2 = 140

t2 = np.array(
    [
        10,
        20,
        30,
        40,
        60,
        80,
        100,
        120,
        150,
        210,
        270,
        330,
        400,
        450,
        645,
        870,
        990,
        1185,
    ]
)
s2 = np.array(
    [
        0.16,
        0.48,
        0.54,
        0.65,
        0.75,
        1.00,
        1.12,
        1.22,
        1.36,
        1.55,
        1.70,
        1.83,
        1.89,
        1.98,
        2.17,
        2.38,
        2.46,
        2.54,
    ]
)
tr2 = t2 / r2**2
s = s2
```

```python

# 设置标准曲线的界限
ymin = 1.0e-2
ymax = 10.0
xmin = 1.0e-1  
xmax = 1.0e4

# 坐标加密可以绘出光滑的曲线
ix = np.linspace(-1, 4, 51)
x = np.float_power(10, ix) # 1/u

# 初始参数，相当于位移 = 0
# T = Q/4/np.pi
# S = 4*T

# 初始参数，按 Jacob 计算
i1 = np.random.randint(0, len(tr2))
i2 = np.random.randint(0, len(tr2))
slope = (s[i1]-s[i2])/np.log10(tr2[i1]/tr2[i2])

T = 0.183*Q/slope
S = 2.25*T*tr2[i1]*np.float_power(10,-s[i1]/slope)
print(T,S)

# Plot the data
def Theis_fit(dlogs, dlogtr2):
    global T, S # 设置全局变量，值可以穿透程序

    
    # 计算参数，配线法不同，公式也不一样
    T = Q/4/np.pi*np.float_power(10, dlogs)
    S = 4*T*np.float_power(10, -dlogtr2)

    # 图形设置
    plt.style.use('default')
    fig, ax = plt.subplots(dpi=100)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_aspect(1)
    plt.xlabel('$\log 1/u$')
    plt.ylabel('$W(u)$')
    ax.grid(True)
    # 绘制标准曲线
    ax.plot(x, exp1(1/x), label="标准曲线")
    # 绘制平移的散点图形
    ax.plot(tr2*np.float_power(10, dlogtr2), s*np.float_power(10, dlogs), '*', 
        label="观测值")
    # 指定图标题，显示中文
    ax.set_title("配线法", fontproperties={'family': 'KaiTi', 'size': 12})
    plt.legend(
        prop={'family': 'Simsun', 'size': 10}, handlelength=6,
        loc=4,title="图例",
        title_fontproperties={'family': 'SimHei', 'size': 12})   
    
    # plt.savefig("out.png") # 保存图片
    
    plt.show()
    # 输出参数
    print('  T = {:.4f} m^2/min'.format(T))
    print('  S = {:.4e}'.format(S))

dlogs_v=np.log10(T*4*np.pi/Q)
dlogs_min=dlogs_v-0.5
dlogs_max=dlogs_v+0.5

dlogs_1 = widgets.FloatSlider(
        value=dlogs_v, min=dlogs_min, max=dlogs_max, step=0.01,
        description="Δ lg(s):",
        continuous_update=False,
        readout_format='.2f',
        disabled=False)

dlogtr2_v=np.log10(4*T/S)
dlogtr2_min=dlogtr2_v-0.5
dlogtr2_max=dlogtr2_v+0.5

dlogtr2_1 = widgets.FloatSlider(
        value=dlogtr2_v, min=dlogtr2_min, max=dlogtr2_max, step=0.01,
        description=r"Δlg(t/r^2):",
        continuous_update=False,
        readout_format='.2f',
        disabled=False)

# ipywidgets 小部件控制参数实现互动。ipywidgets 有缓冲功能，
# 同一个 Notebook 复制的代码得不到所期望的结果
widgets.interact(Theis_fit,dlogs=dlogs_1,dlogtr2=dlogtr2_1,); 
```

**思考题：**

(1) 如何依据不同抽水试验条件选择合理的 Theis 解配线类型和方法？

(2) 为什么实测与标准曲线重合后，匹配点可以任意选取？任意选取匹配点会影响该方法结果的可靠性吗？

```python

```
