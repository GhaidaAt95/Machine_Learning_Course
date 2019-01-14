import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.DataFrame({
    'Age':['20','21','30','20','25','21'],
    'year':[2020,2019,2010,2020,2015,2019],
    'X':[5,3,4,5,5,9]
})
print(df)
print(pd.get_dummies(df))
