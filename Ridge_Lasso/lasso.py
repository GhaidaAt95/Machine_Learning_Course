# Study code learned from https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/#one

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.linear_model import Lasso

def lasso_regression(data, predictors, alpha, models_to_plot={}):
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter = 1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])

    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title("plot for alpha: %.3g"%alpha)

    rss = sum( (y_pred - data['y'])**2 )
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret    


def lasso_regression2(data, predictors, alpha, max_iter, models_to_plot={}):
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter = max_iter)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])

    if max_iter in models_to_plot:
        plt.subplot(models_to_plot[max_iter])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title("plot for max_iter : %.5g "%max_iter)

    rss = sum( (y_pred - data['y'])**2 )
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret    