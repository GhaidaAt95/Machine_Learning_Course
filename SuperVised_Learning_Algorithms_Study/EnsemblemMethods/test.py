from sklearn.model_selection import KFold
import numpy as np

X = np.array([[1,2], [3,4], [5,6], [7,8]])
y = np.array([1,2,3,4])
kf = KFold(2)
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    print("split {}".format(i))
    print("Fold xtrain = {}, ytrain = {}".format(train_idx, train_idx))
    print("Fold xtest = {}, ytest = {}".format(test_idx, test_idx))

