import numpy as np
import matplotlib.pyplot as plt

#闭式解并预测房价

def data():
    X = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
    x_p = np.array(X)
    Y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
    y_p = np.array(Y)
    return x_p,y_p

def close_form(x,y):
    x = np.array([x])
    one = np.ones((1,14))
    vx = np.concatenate([one,x])
    theta = np.dot(np.dot(np.linalg.pinv( np.dot(vx,vx.T) ),vx),y.T)
    print(theta)
    theta0 = theta[0]
    theta1 = theta[1]
    y = x[0] * theta1 + theta0

    print("打印数据：")
    print("y=%fx+%f"%(theta1,theta0))
    print("the housing price in 2014 is %f"%(2014*theta1 + theta0) )

    plt.title('Close Form')
    plt.xlabel('years')
    plt.ylabel('prices')
    plt.scatter(x[0], y, c='#FF0000')
    plt.plot(x[0], y)
    plt.show()

if __name__ == "__main__":
    print("-----------------close form-------------------")
    X,Y = data()
    close_form(X,Y) 
