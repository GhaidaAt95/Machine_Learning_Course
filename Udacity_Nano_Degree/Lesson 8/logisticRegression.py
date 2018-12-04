import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('class_data.csv')
X = np.array(data[['x1','x2']])
y = np.array(data['y'])

classifier = LogisticRegression()
classifier.fit(X,y)

print(type(classifier))

colors = np.where(data['y']==1,'b','r')
plt.scatter(data.x1, data.x2, c=colors)
classifier.plot()
plt.show()