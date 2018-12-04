import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('class_data.csv')

#print(data.head())
colors = np.where(data['y']==1,'b','r')
plt.scatter(data.x1, data.x2, c=colors)
plt.show()

X = np.array(data[['x1','x2']])
Y = np.array(data['y'])