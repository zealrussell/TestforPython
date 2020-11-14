import numpy as np
import matplotlib.pyplot as plt
import random

def load_data():
    file = open("ex4x.dat")
    data, label = [], []
    for line in file.readlines():
        lineArr = line.strip().split()
        data.append([float(lineArr[0] ), float( lineArr[1])] )
    for i in range(80):
        if i < 40 :
            label.append(1)
        else :
            label.append(-1)
    xmat = np.array(data)
    ymat = np.array(label)
    return xmat,ymat

def fsign(x, w, b):
    y = np.dot(x, w) + b
    return y

def fit(X_train, y_train):
    is_wrong = False
    w = np.zeros((2,1))
    b = 0
    l_rate = 0.1
    while not is_wrong:
        wrong_count = 0
        for d in range(len(X_train)):
            X = X_train[d]
            y = y_train[d]
            if y * sign(X, w, b) <= 0:
                w = w + l_rate * np.dot(y, X) # 更新权重
                b = b + l_rate * y # 更新步长
                wrong_count += 1
        if wrong_count == 0:
            is_wrong = True        
    return w,b

def per(X,Y):
    learning_rate = 100
    iterations = 10000

    weight = [0, 0]  # 初始化weight,b
    b = 0
    index = 0   
    for i in range(iterations):
        index = random.randint(0,79)
        x = X[index]
        y = Y[index]
        if y * (weight[0] * x[0] + weight[1] * x[1] + b) <= 0 :
            weight[0] += learning_rate * y * x[0]  # 更新权重
            weight[1] += learning_rate * y * x[1]
            b += learning_rate * y  # 更新偏置量
            #print(weight[0], weight[1], b)
    return weight,b

if __name__ == "__main__":
    X,Y = load_data()
    weight,b = per(X,Y)
    #weight,b = fit(X,Y)

    plotx1 = np.arange(20,60,1)
    plotx2 = []
    for i in range(len(plotx1)):
        plotx2.append( ( (-weight[0] * plotx1[i]- b)/weight[1] )  )
    plt.plot(plotx1,plotx2,c='r',label='decision boundary')

    plt.scatter(X[:,0][Y==-1],X[:,1][Y==-1],marker='o',label='label=neg')
    plt.scatter(X[:,0][Y==1],X[:,1][Y==1],marker='+',label='lebel=pos')
    plt.grid()
    plt.legend()
    plt.show()