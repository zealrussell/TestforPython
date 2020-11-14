import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot

#梯度下降

#归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
    
X = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
Y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]   
a = 0
b = 0
iterations = 500                        #迭代次数
alpha = 0.000045                        #学习率

X = np.array(X)
Y = np.array(Y)
X = normalization(X)                    #归一化处理
Y = normalization(Y)

for i in range(iterations):
    tempa=0
    tempb=0
    for j in range(14):
        tempb += ( (a * X[j] + b - Y[j])) / 14
        tempa += ( (a * X[j] + b - Y[j]) * X[j])/14
    b = b - alpha * tempb
    a = a - alpha * tempa

print("a的值为%f" %a)
print("b的值为%f" %b)

y=a*2014+b
#归一化还原
y= y*(np.max(Y) - np.min(Y) ) + np.min(Y)   #归一化还原
print("2014年房价:%f"%y)


plt.title("linear_regression")
plt.xlabel('x')
plt.ylabel('h(x)')
plt.scatter(X, Y,c="#FF0000")   #打印数据点

plt.show()