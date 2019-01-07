import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

data = np.asarray(pd.read_csv("data.csv", header=None))
X = data[:,0:2]
y = data[:,2]

print(X[0:10,0:2])
print(y[:10])
print(np.argwhere(y==0).flatten())