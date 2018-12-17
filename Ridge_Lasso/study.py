# Study code learned from https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/#one

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.linear_model import LinearRegression
import ridge as R
import lasso as L

rcParams['figure.figsize'] = 12, 10

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)
y = np.sin(x) + np.random.normal(0,0.15,len(x))

data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
print(data.head())
# plt.plot(data['x'],data['y'],'.')
# plt.show()

# Add 14 features --> Tota of 15 features
for i in range(2,16):
    colname = 'x_%d'%i
    data[colname] = data['x']**i

print(data.head())

def linea_regression(data,power, models_to_plot):
    predictors = ['x']
    if power >= 2:
        predictors.extend([
            'x_%d'%i for i in range(2,power+1)
        ])
    # print("HERE*********************************1")
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    # print("HERE*********************************2")

    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('plot for power: %d'%power)
    # print("HERE*********************************3")
    rss = sum( ( y_pred - data['y'] ) **2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    # print("HERE*********************************4")
    return ret

def simple_reg():
    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['model_pow_%d'%i for i in range(1,16)]
    coef_matrix_simple = pd.DataFrame(index=ind, columns=col)
    # print(coef_matrix_simple.head())

    models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

    for i in range(1,16):
        coef_matrix_simple.iloc[i-1,0:i+2] = linea_regression(data=data, power=i, models_to_plot=models_to_plot)
        # print(coef_matrix_simple.head())

    pd.options.display.float_format = '{:.2g}'.format
    print(coef_matrix_simple)


def ridge_reg():
    predictors = ['x']
    predictors.extend(['x_%d'%i for i in range(2,16) ])

    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
    coef_matrix_ridge = pd.DataFrame(index=ind,columns=col)

    models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}

    for i in range(10):
        coef_matrix_ridge.iloc[i,] = R.ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
    
    pd.options.display.float_format = '{:.2g}'.format
    print(coef_matrix_ridge)
    print("*************** Number of Zero Coeff")
    print(coef_matrix_ridge.apply(lambda x: sum(x.values==0),axis=1))


def lasso_reg():
    predictors = ['x']
    predictors.extend(['x_%d'%i for i in range(2,16) ])

    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
    coef_matrix_lasso = pd.DataFrame(index=ind,columns=col)

    models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}
    for i in range(10):
        coef_matrix_lasso.iloc[i,] = L.lasso_regression(data=data,predictors=predictors,alpha=alpha_lasso[i],models_to_plot=models_to_plot)

    pd.options.display.float_format = '{:.2g}'.format
    print(coef_matrix_lasso)
    print("*************** Number of Zero Coeff")
    print(coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1))

def check_iter_lasso():
    predictors = ['x']
    predictors.extend(['x_%d'%i for i in range(2,16) ])

    alpha_lasso = 1e-4
    max_iter = [1e5, 1e4, 1e3, 1e2, 1e-2, 1e-5]

    col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
    ind = ['max_iter_%d'%max_iter[i] for i in range(0,6)]
    coef_matrix_lasso = pd.DataFrame(index=ind,columns=col)
    models_to_plot = {1e5:231, 1e4:232,1e3:233, 1e2:234, 1e-2:235,1e-5:236}

    for i in range(6):
        coef_matrix_lasso.iloc[i,] = L.lasso_regression2(data=data,predictors=predictors,alpha=alpha_lasso,models_to_plot=models_to_plot,max_iter=max_iter[i])

    pd.options.display.float_format = '{:.2g}'.format
    print(coef_matrix_lasso)
    coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)   




# simple_reg()
# ridge_reg()
lasso_reg()

plt.figure(2)
rcParams['figure.figsize'] = 12, 10
check_iter_lasso()

plt.show()
"""
    * Ridge & Lasso help in achiving a parsimonious model 
        (accomplishes desired level of prediction/explanationwith as few features/predictors as possible)
    * 
"""