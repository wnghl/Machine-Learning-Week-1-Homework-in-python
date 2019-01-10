import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from computeCost import *
from gradientDescent import *
from warmUpExercise import *
# ==================== Part 1: Basic Function ====================
print("Running warmUpExercise ... ")
print("5×5 Identity Matrix: ")
warmUpExercise()
print("Program paused. Press enter to continue.")
input()
# ======================= Part 2: Plotting =======================
print("Plotting Data ...")
path = 'ex1data1.txt'
#names添加列名，header用指定的行作为标题，若原来无标题且指定标题则设为None
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
data.head()#观察千前五行是否正确
data.describe()#查看data
data.plot(kind = 'scatter', x = 'Population', y = 'Profit', figsize = (8, 5))
plt.show()

# =================== Part 3: Cost and Gradient descent ===================
#Gradient Descent的一些设置
alpha = 0.01
iterations = 1500
#在训练集中插入一列 x0
data.insert(0, 'X0', 1)
#设置训练集X以及y
cols = data.shape[1]#shape返回一个元祖，为矩阵尺寸，此处取出其列数
X = data.iloc[:, 0 : cols - 1]#利用iloc函数提取行数据
y = data.iloc[:, cols - 1 : cols]#同上
#将X, y转化为numpy矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
#theta是一个1×2矩阵
theta = np.matrix([0, 0])
#计算初始代价函数的值
J = computeCost(X, y, theta)
print("With theta = [0 ; 0]\nCost computed = %f" % J)
print("Expected cost value (approx) 32.07\n")
#进一步测试cost function
J = computeCost(X, y, np.matrix([-1, 2]))
print("\nWith theta = [-1; 2]\nCost computed = %f" % J)
print("Expectexd cost value (approx) 54.24")
#进行gradient descent
print("Running Gradient Descent ... ")

final_theta, cost = gradientDescent(X,  y, theta, alpha, iterations)
#print theta to screen
print("Theta found by gradient descent:")
print(final_theta)
print("Expected theta values (approx)")
print(" -3.6303\n  1.1664\n")
#绘制线性模型和数据，观看其拟合
x = np.linspace(data.Population.min(), data.Population.max(), 100)#横坐标
f = final_theta[0, 0] + (final_theta[0, 1] * x)

fig, ax = plt.subplots(figsize = (6, 4))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(data['Population'], data.Profit, label = "Training Data")
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

print('Visualizing J(theta_0, theta_1) ...\n')
#plot J(theta)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(iterations), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
