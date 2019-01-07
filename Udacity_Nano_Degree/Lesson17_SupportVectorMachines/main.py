import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def plot_classificatio_2D(X,y,model):
    x_min, x_max = min(X[:,0]) -0.1 , max(X[:,0]) +0.1
    y_min, y_max = min(X[:,1]) -0.1 , max(X[:,1]) +0.1

    xx, yy = np.meshgrid(np.arange(x_min,x_max,0.001),
                        np.arange(y_min,y_max, 0.001))

    y_pred2 = model.predict(np.c_[xx.ravel(), yy.ravel()])
    y_pred2 = np.reshape(y_pred2, xx.shape)
    plt.contourf(xx, yy, y_pred2, cmap=plt.cm.RdBu.reversed(), levels=2, alpha=0.5)
    plt.contour(xx,yy,y_pred2, colors='k', levels=1, alpha=1, linewidths =1)
    plot_colors = "br"
    for i, color in zip(range(2), plot_colors):
        idx = np.where(y == i)  
        plt.scatter(X[idx,0], X[idx,1], c=color, edgecolor='black',s=30)

    plt.title("Solution Boundry")
    plt.show()

    
'''
    # Hyperparameters:
        1) C - Parameter: it goes with the classification error
            Large - Focus on classification
            Small - Focus on having a large Margin
        2) Kernel :
            - linear 
            - poly : polynomail 
            - rbf : radial basis functions
        3) Degree : if it is polynomial, this is the maximum degree of the monomials in the kenrel
        4) gamma : For when the kernel is rbf
            - Large - Narrow mountins - could cause overfitting
            - Small - Wide mountins - could cause underfitting

'''

df = pd.read_csv('data.csv', header=None)

data = np.asarray(df)

X = data[:,0:2]
y = data[:,2]

model  = SVC(kernel='rbf',gamma=10,C=6)
model.fit(X,y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print("The accuraccy is {}%".format(acc*100))

plot_classificatio_2D(X,y,model)

