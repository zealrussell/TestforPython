import numpy as np
import matplotlib.pyplot as plt

#逻辑回归 牛顿法

#加载数据
def load_data(file):
    data,label = [],[]
    for line in file.readlines():
        lineArr = line.strip().split()
        data.append([1.0, float(lineArr[0] ), float( lineArr[1])] )
    for i in range(80):
        if i < 40 : 
            label.append(1)
        else :
            label.append(0)
    xmat = np.mat(data)
    ymat = np.mat(label).reshape(80,1)
    return xmat,ymat

def H_1(train_x,train_y,theta):#海森矩阵的逆
    train_x["x3"]=[1.0 for i in range(len(train_x))]
    a = np.zeros((3,3))
    for i in range(len(train_x)):
        a = a + np.eye(3,3)*(np.array(np.exp(np.dot(theta,np.mat(train_x.iloc[i, :]).T)))[0][0]/(np.array(np.exp(np.dot(theta,np.mat(train_x.iloc[i, :]).T)))[0][0]+1)**2)* np.mat(train_x.iloc[i, :]).T*np.mat(train_x.iloc[i, :])
    return (a / len(train_x)).I

def eye(X):
    return np.diag()

def w_calc (xmat,ymat,maxIter = 7):
    W = np.zeros((3,1))
    m = xmat.shape[0]                    #80
    for i in range(maxIter):
        h = 1.0/(1+np.exp(-(xmat*W)) )     #(80,1)
        grad = (1.0/m)*xmat.T*(h-ymat)     #(3,1) = (3,80) * (80,1)
        H = (1.0/m)*xmat.T*np.diag((h*(1-h).T).A[0] ).T*xmat  #(3,3) =(3,80)*(80,1)*(1,80)*(80,3)
        W -= np.linalg.inv(H)*grad       #(3,1) = 
    return W

file = open("ex4x.dat")
xmat,ymat = load_data(file)
W = w_calc(xmat,ymat)
print('w:',W)

w0=W[0,0]
w1=W[1,0]
w2=W[2,0]
plotx1 = np.arange(20,50,2)
plotx2 = -w0/w2-w1/w2*plotx1
plt.plot(plotx1,plotx2,c='r',label='decision boundary')

plt.scatter(xmat[:,1][ymat==0].A,xmat[:,2][ymat==0].A,marker='o',label='label=neg')
plt.scatter(xmat[:,1][ymat==1].A,xmat[:,2][ymat==1].A,marker='+',label='lebel=pos')
plt.grid()
plt.legend()
plt.show()
