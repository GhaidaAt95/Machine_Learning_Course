import numpy as np
from sklearn.preprocessing import MinMaxScaler

weights = np.array([[115],[140],[175]], dtype=float)

'''
 Each element of the numoy array is going to be a 
    different training point and then each element within that 
    training point is going to be a feature
'''

scaler = MinMaxScaler()

rescaled_weight = scaler.fit_transform(weights)

print('-'*70)
print("Before: ")
print(weights)
print('-'*70)
print("After: ")
print(rescaled_weight)
