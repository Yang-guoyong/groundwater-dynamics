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

<!-- #region -->
# 拐点图解法


根据 Hantush-Jacob 公式

$$
s=W(u,\beta\,)=\frac{Q}{4\pi T}\int_{u}^{\infty}\frac{1}{y}e^{-y-\frac{\beta^2}{4y}}\,dy
$$


**拐点 $P$ 的性质**

- 坐标与降深：
   $$
   t_p=\frac{SBr}{2T},\quad s_p=\frac{1}{2}s_{max}=\frac{Q}{4\pi T}K_0\left(\frac{r}{B}\right)
   $$
- 切线斜率：
   $$
   i_p=\frac{2.3Q}{4\pi T}e^{-\frac{r}{B}}
   $$
- $s_p$、$i_p$ 的关系：
   $$
   2.3\frac{s_p}{i_p}=e^{\frac{r}{B}}K_0\left(\frac{r}{B} \right)
   $$


**思路**

1. 调整参数确定拐点坐标与降深 $(t_p,s_p)$，最大降深 $s_{max}$用外推法确定;
2. 调整参数确定拐点切线斜率 $i_p$；
3. 二分法求解方程: $e^{\frac{r}{B}}K_0\left(\frac{r}{B} \right)-2.3\frac{s_p}{i_p}=0$:
   $$
   f'(x)=e^{x}(K_0(x)-K_1(x))<0
   $$
   $f(x)$ 在 $[0.01,5]$ 上是单调递减的，二分法求解有效。
4. 计算参数：
   $$
   B=\frac{r}{\left[\frac{r}{B} \right]},\quad T=\frac{2.3Q}{4\pi i_p}e^{-\frac{r}{B}},\quad S=\frac{2Tt_p}{Br}
   $$
5. 验证：避免确定 $s_{max}$ 的随意性, 用 Hantush-Jacob 公式进行验证。
<!-- #endregion -->

**例：**

某河阶地，上部为潜水层，其下为 $2m$ 厚弱透水的亚砂土，再下为 $1.5m$ 厚的中、粗砂层(承压)。水源地抽取承压水，以 T32 号孔做非稳定抽水试验，距它 $197m$ 处有 T31 孔观恻，抽水量为 $Q=69.1m^3/h$，观测孔水位降深值如下表，求取水文地质参数。

<center> 表  某越流含水层抽水试验T31观测孔降深资料

| 抽水累计时间 $t(min)$ | 水位降深 $s(m)$ | 抽水累计时间 $t(min)$ | 水位降深 $s(m)$ | 抽水累计时间 $t(min)$ | 水位降深 $s(m)$ |
| :-------------------: | :-------------: | :-------------------: | :-------------: | :-------------------: | :-------------: |
|           1           |      0.05       |          60           |      0.575      |          300          |      0.763      |
|           4           |      0.054      |          75           |      0.62       |          330          |      0.77       |
|           7           |      0.12       |          90           |      0.64       |          360          |      0.77       |
|          10           |      0.175      |          120          |      0.685      |          390          |      0.785      |
|          15           |      0.26       |          150          |      0.725      |          420          |      0.79       |
|          20           |      0.33       |          180          |      0.735      |          450          |      0.792      |
|          25           |      0.383      |          210          |      0.755      |          480          |      0.794      |
|          30           |      0.425      |          240          |      0.76       |          510          |      0.795      |
|          45           |      0.52       |          270          |      0.76       |          540          |      0.796      |
</center> 

```python
%matplotlib widget

import numpy as np
import math as math
from scipy.optimize import bisect
from scipy.special import k0
import matplotlib.pyplot as plt
import ipywidgets as widgets
from wellfunction import *

# 控制小数的显示精度
np.set_printoptions(precision=4)

plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号
```

```python
# 准备数据
Q = 69.1/60  # m^3/min
r = 197.0    # m

# min
t = np.array([
    1, 4, 7, 10, 15, 20, 25, 30, 45, 60, \
    75, 90, 120, 150, 180, 210, 240, 270, 300, 330, \
    360, 390, 420, 450, 480, 510, 540])
# m
s = np.array([
    0.05, 0.054, 0.12, 0.175, 0.26, 0.33, 0.383, 0.425, 0.52, \
    0.575, 0.62, 0.64, 0.685, 0.725, 0.735, 0.755, 0.76, 0.76, \
    0.763, 0.77, 0.772, 0.785, 0.79, 0.792, 0.794, 0.795, 
    0.796])
n = len(t)

# 计算绘图范围
imin = math.floor(math.log10(min(t)))  # math.floor(x)返回小于x的最大整数
imax = math.ceil(math.log10(max(t)))   # math.ceil(x)返回大于等于x的最小整数               
xmin = 10**imin
xmax =  10**imax

ymin = 0.0
ymax = math.ceil(max(s*10))/10

# 绘函数图像的网格
x = np.linspace(imin, imax, (imax-imin)*10+1)
x = np.float_power(10,x)


# 设定初始的 sp, tp, i
sp = 0.5*s[-1]
tp = t[math.floor(n/2)]
slope = 0.5

# 设置初始的 T, S, beta
T = 1
S = 1.0e-4
B = 1.0e+4
```

```python
# Plot the data
def inflection_point_fit(tp, sp, slope):
    
    global T, S, B   # 这些都是全局变量，改变后函数外的值同时改变
    
    plt.style.use('default')  # 绘图风格
    
    fig = plt.figure()
    ax = fig.add_subplot()
                     
    '''
    若不想设置其他内容，也可合并写为
    fig, ax = plt.subplots(figsize=(6, 4))
    '''                 

    ax.plot(t, s, '*', label="观测值")
    ax.plot(x, sp + slope*np.log10(x/tp), label="拐点切线")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    plt.xlabel('$\log t$')
    plt.ylabel('$s$')
    ax.grid(True)

    ax.set_title("拐点法", fontproperties={'family': 'KaiTi', 'size': 12}) # 指定显示中文字体
    
    # 调用 scipy.optimize.bisect(二分法)
    beta = bisect(lambda x:np.exp(x)*k0(x)-2.3*sp/slope,0.01,5)
    
    # 计算参数
    T = 2.3*Q*np.exp(-beta)/4/np.pi/slope
    S = 2*T*tp*beta/r**2
    B = r/beta
    
    # 绘图数据
    u = 0.25*r**2*S/T/x
    ax.plot(x, 0.25*Q*hantush_jacob(u, beta)/np.pi/T, label="标准曲线")
    
    plt.legend(
        prop={'family': 'Simsun', 'size': 10}, handlelength=2,
        loc=4,title="图例",
        title_fontproperties={'family': 'KaiTi', 'size': 12})

    plt.show()
    print('             T(m^2/min): ', '{:.4f}'.format(T))
    print('                      S: ', '{:.4e}'.format(S))
    print('                   B(m): ', '{:.4e}'.format(B))    
```

```python
widgets.interact(
    inflection_point_fit,
    tp = widgets.FloatSlider(
        value=tp, min=1, max=0.75*t[-1], step=1,
        description=r'$t_p$ [-]:', continuous_update=False,
        readout_format='.1f', disabled=False),
    sp = widgets.FloatText(
        value=sp, description=r'$s_p$ [-]:', 
        continuous_update=False, disabled=False),
    slope = widgets.FloatSlider(
        value=slope, min=0.1, max=1, step=0.01,
        description=r'$slope$ [-]:', continuous_update=False,
        readout_format='.3f', disabled=False)
    );
```

```python
print('             T(m^2/min): ', '{:.4f}'.format(T))
print('                      S: ', '{:.4e}'.format(S))
print('                   B(m): ', '{:.4e}'.format(B))   
```

进一步改进：如何通过最大降深快速确定拐点？



有些时候 matplotlib 的绘图没法显示在 notebook 中，或者显示不了。这与 backend 有关。


**例：作业 6**

对某承压含水层抽水孔进行12h的定流量非稳定流抽水，抽水量为 $528m ^3 /d$，距抽水孔 90m  处有一观测孔，观测数据如下表 4，试确定水文地质参数（直线图解）。

<center> 表 4 承压含水层抽水试验观测数据

| 序号 | 时间(min) | 降深(cm) | 序号 | 时间(min) | 降深(cm) |
| :--: | :-------: | :------: | :--: | :-------: | :------: |
|  1   |     1     |   2.5    |  9   |    50     |   24.7   |
|  2   |     2     |   3.9    |  10  |    60     |   26.4   |
|  3   |     4     |   6.1    |  11  |    90     |   30.4   |
|  4   |     6     |   8.0    |  12  |    120    |   33.0   |
|  5   |     9     |   10.6   |  13  |    150    |   35.0   |
|  6   |    20     |   16.8   |  14  |    360    |   42.6   |
|  7   |    30     |   20.0   |  16  |    550    |   44.0   |
|  8   |    40     |   22.6   |  16  |    720    |   44.5   |

```python

```
