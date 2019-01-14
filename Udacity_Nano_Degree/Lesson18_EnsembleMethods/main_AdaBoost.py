import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

'''
    HyperParameters:
        1) Base_Estimator 
            Model utilized for the weak learners
        2)n_estimators: 
            Max # of weak learners used0n  
'''