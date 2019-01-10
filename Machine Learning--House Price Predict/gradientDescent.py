import numpy as np
import pandas as pd
from computeCost import *
def gradientDescent(X, y, theta, alpha, iterations):
    """注意函数两个返回值 theta, cost"""

    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.flatten().shape[1])#求出参数theta的数量
    cost = np.zeros(iterations)#初始化一个numpy array，包含每次迭代的cost
    m = X.shape[0]

    for i in range(iterations):
        #向量化方法进行gradient descent, 可以大大提高效率
        temp = theta - (alpha / m) * (X * theta.T - y).T * X
        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
