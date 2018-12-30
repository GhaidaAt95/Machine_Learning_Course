import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(42)

'''
    # y_prediction = step(w_1*x_1 + w_2*x_2 + b)
'''
def stepFunction(t):
    if t >= 0:
        return 1
    else:
        return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


def percptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_pred = prediction(X[i],W,b)
        # a positive point in a negative area
        if y[i] - y_pred == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_pred == -1:
            # negative point in a positive area 
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_ephocs = 25):
        x_min, x_max = min(X.T[0]), max(X.T[0])
        y_min, y_max = min(X.T[1]), max(X.T[1])

        W = np.array(np.random.rand(2,1))
        b = np.random.rand(1)[0] + x_max

        boundary_lines = []

        for i in range(num_ephocs):
            W, b = percptronStep(X, y, W, b, learn_rate)

            boundary_lines.append((-W[0]/W[1], -b/W[1]))
        
        return boundary_lines

def readAndPassData(name):
    df = pd.read_csv(name, header=None)
    X = df.iloc[:,:2].values
    y = df.iloc[:,2].values

    x_min, x_max = min(X.T[0]), max(X.T[0])
    print("x_min: {} x_max: {}".format(x_min,x_max))
    y_min, y_max = min(X.T[1]), max(X.T[1])
    print("y_min: {} y_max: {}".format(y_min,y_max))

    boundary_lines = trainPerceptronAlgorithm(X, y)

    colors = np.where(df.iloc[:,2] == 1, 'b','r')
    plt.ylim([-0.5, 1.5])
    plt.xlim([-0.5, 1.5])
    plt.figure(1)
    plt.scatter(x=X.T[0],y=X.T[1],c = colors, linewidths=1,edgecolors='k')
    x = np.asarray(X.T[0])
    m = np.asarray(boundary_lines)
    for bLine in m[:len(m)-1]:
        y_plt = x * bLine[0] +  bLine[1]  
        plt.plot(x,y_plt,'g','--',dashes=[0.05,0.1],linewidth=0.1,)
    y_plty = x * m[len(m)-1][0] +   m[len(m)-1][1] 
    plt.plot(x,y_plt,'k') 
    plt.show()
readAndPassData("data.csv")