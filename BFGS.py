'''
    Реализация алгоритма BFGS на Python/Numpy/Scipy.
'''

import numpy as np
import numpy.linalg as ln
import scipy as sp
import time
import psutil

from utils import f, f_der

# Реализация BFGS
def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
    """
    Минимизация функции f, используя алгоритм BFGS.
    
    Параметры
    ----------
    f : f(x)
        Минимизируемая функция.
    x0 : ndarray
        Начальная точка.
    fprime : fprime(x)
        Градиент функции f.
    """
    
    if maxiter is None:
        maxiter = len(x0) * 10000

    # Начальные значения
    k = 0
    gfk = fprime(x0)
    N = len(x0)
    # Создаем единичную матрицу I
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
   
    while ln.norm(gfk) > epsi and k < maxiter:
        
        # pk - направление поиска
        
        pk = -np.dot(Hk, gfk)
        
        # line_search не только alpha
        # но только это значение интересно для нас

        line_search = sp.optimize.line_search(f, f_der, xk, pk)
        alpha_k = line_search[0]

        # breakpoint()
        
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        # print(f'Итерация: {k}; найденная точка x: {xk}')
        
        gfkp1 = fprime(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        
        k += 1
        
        ro = 1.0 / (np.dot(yk, sk))
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
        
    return (xk, k)


def main():
    ordinary_F()


def ordinary_F():
    start_time = time.time()  # время начала выполнения

    minimum_x, iter_count = bfgs_method(f, f_der, np.array([0.1, 0.1]))
    minimum_value = f(minimum_x)

    print(f'Минимум функции f(x) = {minimum_value}, достигается в точке x = {minimum_x}')
    print(f'Количество итераций до нахождения решения: {iter_count}')

    end_time = time.time()  # время окончания выполнения
    execution_time = end_time - start_time  # вычисляем время выполнения
    
    print(f"Время выполнения программы: {execution_time} секунд")
    print(f"Память: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} МБ")

if __name__ == "__main__":
    main()