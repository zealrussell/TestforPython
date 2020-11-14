import numpy as np
import matplotlib.pyplot as plt

#逻辑回归梯度下降法


#加载数据
def load_data(file):
    data,label = [],[]
    for line in file.readlines():
        lineArr=line.strip().split()
        data.append([1.0, float(lineArr[0] ), float( lineArr[1])] )
    for i in range(80):
        if i < 40 : 
            label.append(1)
        else :
            label.append(0)
    xmat = np.asmatrix(data)
    ymat = np.asmatrix(label).T  #(80,1)
    return xmat,ymat

#计算
def w_calc (xmat,ymat,alpha = 0.01,maxIter = 1001):
    W = np.asmatrix(np.zeros((3,1)) )
    for i in range(maxIter):
        H = 1/(1+np.exp(-xmat*W))   #(80,1)
        dw = xmat.T*(H-ymat) #(3,1) = (3,80)*(80,1)
        W += alpha * dw
    return W

file = open("ex4x.dat")
xmat,ymat = load_data(file)
W = w_calc(xmat,ymat)
print('w:',W)

w0=W[0,0]
w1=W[1,0]
w2=W[2,0]

#图形化显示
plotx1 = np.arange(20,60,1)
plotx2 = -w0/w2-w1/w2*plotx1
plt.plot(plotx1,plotx2,c='r',label='decision boundary')

plt.scatter(xmat[:, 1][ymat ==0].A,xmat[:,2][ymat==0].A,marker='o',label='label=neg')
plt.scatter(xmat[:, 1][ymat==1].A,xmat[:,2][ymat==1].A,marker='+',label='lebel=pos')
plt.grid()
plt.legend()
plt.show()