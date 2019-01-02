import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

'''
    Hyperparameters:
        1. Max_depth
        2. min_samples_leaf
        3. min_samples_split
        4. max_features : The # of features to consider when looking for the best split
'''

# Read Data
data = np.asarray(pd.read_csv('data.csv',header=None))

X = data[:,:2]
y = data[:,2]

model = DecisionTreeClassifier()
model.fit(X,y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)

print(acc)


x_min, x_max = min(X[:,0]) -0.1 , max(X[:,0]) +0.1
y_min, y_max = min(X[:,1]) -0.1 , max(X[:,1]) +0.1

xx, yy = np.meshgrid(np.arange(x_min,x_max,0.001),
                     np.arange(y_min,y_max, 0.001))

# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
y_pred = np.reshape(y_pred, xx.shape)
plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdBu.reversed(), levels=2, alpha=0.5)
plt.contour(xx,yy,y_pred, colors='k', levels=1, alpha=1, linewidths =1)
plot_colors = "br"
for i, color in zip(range(2), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx,0], X[idx,1], c=color, edgecolor='black',s=30)

plt.title("Solution Boundry")
# plt.scatter(X[:,0],X[:,1], c=y, s=20, edgecolor='k')
print("The accuraccy is {}%".format(acc*100))

plt.show()
