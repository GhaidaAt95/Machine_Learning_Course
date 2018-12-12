import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import scipy.stats as stats
# import visuals as vs

# Load The Boston Housing DataSet
data = pd.read_csv("housing.csv")
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

print("Boston Housing dataset has {} data points with {} variables each".format(*data.shape))
print(prices.describe())
print(prices.min())

# plt.bar(features['PTRATIO'],prices,align='center')
# plt.show()

