'''
    Реализация алгоритма Gradient Descend на Python/Numpy.
'''

import numpy as np
import numpy.linalg as ln
import time

from utils import f, f_der
import psutil

def gradient_descent(fprime, x0, lr, maxiter, epsi=10e-3):
    
    """
    Минимизация функции f, используя алгоритм Gradien descend.
    
    Параметры
    ----------
    x0 : ndarray
        Начальная точка.
    fprime : fprime(x)
        Градиент функции f.
    lr: learning_rate.
        Cкорость обучения
    """

    if maxiter is None:
        maxiter = len(x0) * 200

    # Начальные значения
    x = x0
    gfk = fprime(x0)
    k = 0

    # for i in range(max_iterations):
    while ln.norm(gfk) > epsi and k < maxiter:
        gradient = fprime(x)
        x = x - lr * gradient

        gfkp1 = fprime(x)
        gfk = gfkp1
        
        k += 1

        # print(f'Итерация: {k}; найденная точка x: {x}')

    return x, k


def main():
    ordinary_F()


def ordinary_F():
    start_time = time.time()  # время начала выполнения

    # Гиперпараметры
    x0 = np.array([24, -99])
    lr = 0.5
    maxiter = 100

    minimum_x, iter_count = gradient_descent(f_der, x0, lr, maxiter)
    minimum_value = f(minimum_x)

    print(f'Минимум функции f(x) =, {minimum_value}, достигается в точке x = {minimum_x}')
    print(f'Количество итераций до нахождения решения: {iter_count}')

    end_time = time.time()  # время окончания выполнения
    execution_time = end_time - start_time  # вычисляем время выполнения
    
    print(f"Время выполнения программы: {execution_time} секунд")
    print(f"Память: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} МБ")


if __name__ == "__main__":
    main()