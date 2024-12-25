import numpy as np

# # Целевая функция
# def f(x):
#     return x[0]**2 - x[0]*x[1] + x[1]**2 + 9*x[0] - 6*x[1] + 20

# # Производная
# def f_der(x):
#     return np.array([2 * x[0] - x[1] + 9, -x[0] + 2*x[1] - 6])


def f(x):
    """The Rosenbrock function from https://habr.com/ru/articles/439288/"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0, axis=0)

def f_der (x):
    """The Rosenbrock function from https://habr.com/ru/articles/439288/"""
    xm = x [1: -1]
    xm_m1 = x [: - 2]
    xm_p1 = x [2:]
    der = np.zeros_like (x)
    der [1: -1] = 200 * (xm-xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1-xm)
    der [0] = -400 * x [0] * (x [1] -x [0] ** 2) - 2 * (1-x [0])
    der [-1] = 200 * (x [-1] -x [-2] ** 2)
    return der