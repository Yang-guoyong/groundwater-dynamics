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

```python
from math import *
import numpy as np
from numba import njit

# Modified Bessel function K_0
#jit(nopython=True)
@njit
def besselk0(x):
    if x <= 2.0:
        y = x*x/4.0
        besselk0 = (-log(x/2.)*besseli0(x) - 0.57721566 \
                    + y*(0.4227842 + y*(0.23069756 \
                    + y*(0.348859e-1 + y*(0.262698e-2 \
                    + y*(0.1075e-3 + y*0.74e-5))))))
    else:
        y = (2.0/x)
        besselk0 = (exp(-x)/sqrt(x)*(1.25331414 \
                    + y*(-0.7832358e-1 + y*(0.2189568e-1 \
                    + y*(-0.1062446e-1 + y*(0.587872e-2 \
                    + y*(-0.25154e-2 + y*0.53208e-3)))))))
    return besselk0

# Modified Bessel function K_1
@njit
def besselk1(x):
    if x <= 2.0:
        y = x*x/4.0
        besselk1 = (log(x/2.)*besseli1(x) \
                    + (1./x)*(1. + y*(0.15443144 + y*(-0.67278579 \
                    + y*(-0.18156897 + y*(-0.1919402e-1 \
                    + y*(-0.110404e-2 + y*(-0.4686e-4))))))))
    else:
        y = 2.0/x
        besselk1 = (exp(-x)/sqrt(x)*(1.25331414 \
                    + y*(0.23498619 + y*(-0.3655620e-1 \
                    + y*(0.1504268e-1 + y*(-0.780353e-2 \
                    + y*(0.325614e-2 + y*(-0.68245e-3))))))))
    return besselk1


#  Modified Bessel function I_0
@njit
def besseli0(x):
    if x < 3.75:
        y = (x/3.75)**2
        besseli0 = (1.+ y*(3.5156229 + y*(3.0899424 \
                    + y*(1.2067492 + y*(0.2659732 \
                    + y*(0.360768e-1 + y*0.45813e-2))))))
    else:
        y = 3.75/x
        besseli0 = (exp(x)/sqrt(x)*(0.39894228 \
                    + y*(0.1328592e-1 + y*(0.225319e-2 \
                    + y*(-0.157565e-2 + y*(0.916281e-2 \
                    + y*(-0.2057706e-1 + y*(0.2635537e-1 \
                    + y*(-0.1647633e-1 + y*0.392377e-2)))))))))
    return besseli0


# Modified Bessel function I_1
@njit
def besseli1(x):
    if x < 3.75:
        y = (x/3.75)**2
        besseli1 = (x*(0.5e0 + y*(0.87890594 \
                    + y*(0.51498869 + y*(0.15084934 \
                    + y*(0.2658733e-1 + y*(0.301532e-2 \
                    + y*0.32411e-3)))))))
    else:
        y = 3.75/x
        besseli1 = (exp(x)/sqrt(x)*(0.39894228 \
                    + y*(-0.3988024e-1 + y*(-0.362018e-2 \
                    + y*(0.163801e-2 + y*(-0.1031555e-1 \
                    + y*(0.2282967e-1 + y*(-0.2895312e-1 \
                    + y*(0.1787654e-1 + y*(-0.420059e-2))))))))))
    return besseli1


# Bessel function J_0
@njit
def besselj0(x):
    if abs(x) < 8.:
        y = x**2
        besselj0 = (
            (57568490574.+y*(-13362590354.+y*(651619640.7 +
            y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))))) /
            (57568490411.+y*(1029532985.+y*(9494680.718+y *
            (59272.64853+y*(267.8532712+y)))))
        )
    else:
        ax = abs(x)
        z = 8.0/ax
        y = z**2
        xx = ax-.785398164
        besselj0 = (
            sqrt(.636619772/ax)*(cos(xx)*(1.+y *
            (-.1098628627e-2+y*(.2734510407e-4+y * (-.2073370639e-5 +
              y*.2093887211e-6))))-z*sin(xx)*(-.1562499995e-1+y *
              (.1430488765e-3+y*(-.6911147651e-5+y*(.7621095161e-6+y
            * (-.934945152e-7))))))     )
    return besselj0


# Bessel function J_1
@njit
def besselj1(x):
    if abs(x) < 8.:
        y = x**2
        besselj1 = (x*(72362614232. \
                    + y*(-7895059235. + y*(242396853.1 \
                    + y*(-2972611.439 + y*(15704.48260 \
                    + y*(-30.16036606))))))/(144725228442. \
                    + y*(2300535178. + y*(18583304.74 \
                    + y*(99447.43394+y*(376.9991397+y))))))
    else:
        ax = abs(x)
        z = 8.0/ax
        y = z**2
        xx = ax-2.356194491
        besselj1 = (sqrt(.636619772/ax)*(cos(xx)*(1. \
                    + y*(.183105e-2 + y*(-.3516396496e-4 \
                    + y*(.2457520174e-5 + y*(-.240337019e-6))))) \
                    - z*sin(xx)*(.04687499995 + y*(-.2002690873e-3 \
                    + y*(.8449199096e-5 + y*(-.88228987e-6 \
                    + y*.105787412e-6)))))*np.sign(complex(1.0, x)).real)
    return besselj1


# Find a root of BesselK0 near rstart by Newton-Raphson method
@njit
def bj0root(rstart):
    xacc = 1.0e-8
    bj0root = rstart
    for j in range(100):
        dx = -besselj0(bj0root) / besselj1(bj0root)
        bj0root = bj0root - dx
        if abs(dx) < xacc:
            break
    return bj0root


# Calculate Gaussian quadrature nodes and weights
# Source: Numerical Recipes, Press, et al. 1992
def gauleg(x1, x2, n):
    x, w = np.polynomial.legendre.leggauss(n)
    xm = 0.5*(x2 + x1)
    xl = 0.5*(x2 - x1)

    return xm + xl*x[:], xl*w[:]


# Extrapolate a series by epsilon algorithm
@njit
def epsilonn(n, psum):
    eps = np.zeros((202, 201))

    for m in range(n + 1):
        eps[0, m] = psum[m]
    for i in range(1, n + 1):
        for m in range(n - i, -1, -1):
            eps[i, m] = eps[i - 2, m + 1] + 1.0 / \
                (eps[i - 1, m + 1] - eps[i - 1, m])

    return eps[n, 0]


@njit
def func(u, gamma, x, ncase):
    if ncase == 1:
        func = 64.0*x*besselj0(x*np.sqrt(gamma))*summ(x, u, gamma, ncase)
    if ncase == 2:
        func = (4.0*x*besselj0(x*np.sqrt(gamma))*((1.0 \
                - np.exp(-gamma*x*np.tanh(x)/(4.0*u)))*np.tanh(x)/(2.0*x**3) \
                + summ(x, u, gamma, ncase)))

    return func


@njit
def summ(x, u, gamma, ncase):
    nsum = 12
    psum = np.zeros(201)

    for m in range(1, nsum + 1):
        a = 4.0*x**2 + (2.0*m - 1.0)**2*np.pi**2
        denom = (2.0*m - 1.0)**2*np.pi**2*a
        if ncase == 1:
            dsum = (1 - np.exp(-gamma/16.0/u*a))/denom
        if ncase == 2:
            dsum = 16.0/denom
        psum[m] = psum[m - 1] + dsum

#    return  epsilonn(nsum, psum)
    return psum[nsum]


# Neuman's type A and B unconfined aquifer well function
def neumanw(u, gamma, ncase):
    r = np.zeros(201)

    # Define integration ranges using even roots of J0
    nr = 200
    r[0] = 0.0
    r[1] = 5.520078110080565

    for i in range(2, nr+1):
        r[i] = bj0root(r[i - 1] + 2*np.pi)

    ng = 30
    x = np.zeros(30)
    w = np.zeros(30)
    wsum = np.zeros(201)
    gammasq = np.sqrt(gamma)
    wsum[0] = 0.0

    # Integrate by subintervals
    for j in range(1, nr+1):
        x, w = gauleg(r[j - 1] / gammasq, r[j] / gammasq, ng)
        dwf = 0.0

        # Perform Gaussian quadrature
        for i in range(ng):
            dwf = dwf + w[i] * func(u, gamma, x[i], ncase)

        wsum[j] = wsum[j - 1] + dwf

    # Use epsilon algorithm to extrapolate
    return epsilonn(nr, wsum)
```

```python
def timer(func):   # timer 装饰器测试时间
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.6f s' % (func.__name__, time_spend))
        return result
    return func_wrapper
```

```python
@timer
def Neumann(ua, uy, gamma):
    n = len(gamma)

    m = len(ua)
    value = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            value[i, j] = neumanw(ua[i], gamma[j], 1)
            
    np.savetxt('w_ua.csv', value, delimiter=',')

    m = len(uy)
    value = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            value[i, j] = neumanw(uy[i], gamma[j], 2)
    
    np.savetxt('w_uy.csv', value, delimiter=',')
    
    return "Data has been written to file: w_ua.csv and w_uy.csv."
```

```python
ua = np.array([
    2.50E+00, 1.25E+00, 7.14E-01, 4.17E-01, 2.50E-01,
    1.25E-01, 7.14E-02, 4.17E-02, 2.50E-02, 1.25E-02,
    7.14E-03, 4.17E-03, 2.50E-03, 1.25E-03, 7.14E-04,
    4.17E-04, 2.50E-04, 1.25E-04, 7.14E-05, 4.17E-05])

uy = np.array([
    2.50E+03, 1.25E+03, 7.14E+02, 4.17E+02, 2.50E+02,
    1.25E+02, 7.14E+01, 4.17E+01, 2.50E+01, 1.25E+01,
    7.14E+00, 4.17E+00, 2.50E+00, 1.25E+00, 7.14E-01,
    4.17E-01, 2.50E-01, 1.25E-01, 7.14E-02, 4.17E-02,
    2.50E-02, 1.25E-02, 7.14E-03, 4.17E-03, 2.50E-03])

gamma = np.array([0.001, 0.004, 0.010, 0.030, 0.060,
                  0.100, 0.200, 0.400, 0.600, 0.800,
                  1.000, 1.500, 2.000, 2.500, 3.000,
                  4.000, 5.000, 6.000, 7.000])

Neumann(ua, uy, gamma)
```

```python

```
