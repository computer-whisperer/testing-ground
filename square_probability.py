import scipy
import scipy.integrate
import math


def func_1(a, b, c, d):
    return math.sqrt((a-b)**2 + (b-c)**2)


def func_2(b, c, d):
    return scipy.integrate.quad(func_1, 0, 1, (b, c, d))[0]


def func_3(c, d):
    return scipy.integrate.quad(func_2, 0, 1, (c, d))[0]


def func_4(d):
    return scipy.integrate.quad(func_3, 0, 1, (d))[0]


def func_5():
    return scipy.integrate.quad(func_4, 0, 1)[0]

print(func_5())
