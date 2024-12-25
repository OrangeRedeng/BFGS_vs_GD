import numpy as np

# Целевая функция
def f(x):
    return x[0]**2 - x[0]*x[1] + x[1]**2 + 9*x[0] - 6*x[1] + 20

# Производная
def f_der(x):
    return np.array([2 * x[0] - x[1] + 9, -x[0] + 2*x[1] - 6])