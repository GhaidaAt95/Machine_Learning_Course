import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


def print_prop():
    i = 0
    for item in dir(bmi_life_model):
        if i <2 :
            print(item, end="\t\t\t")
            i+= 1
        else:
            print(item)
            i=0
    print()

def calc_life_expectancy_one_feature():
    # Load Data
    bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
    # print(bmi_life_data.head())

    y_values = bmi_life_data['Life expectancy']
    x_values = bmi_life_data.drop(['Life expectancy','Country'], axis=1)

    # print(y_values.head())
    # print(x_values.head())

    bmi_life_model = LinearRegression().fit(X=x_values,y=y_values)

    loas_life_exp = bmi_life_model.predict([[21.07931]])

    print("The Life Expectancy for a BMI of {} is {}".format(21.07931,loas_life_exp))
    print('*'*70)
    print("Coeffiants : ",bmi_life_model.coef_)
    print("Intercept : ",bmi_life_model.intercept_)

    plt.scatter(x_values,y_values,color='g')
    plt.plot(x_values,bmi_life_model.predict(x_values),color='k')
    plt.show()

def calc_house_price_13features():
    data = load_boston()
    x = data['data']
    y = data['target']

    model = LinearRegression().fit(X=x, y=y)

    sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
    
    prediction = model.predict(sample_house)
    print("Prediction : ",prediction)
    print('*'*70)
    print("Coeffient : ",model.coef_)
    print("Intercept : ",model.intercept_)
    plt.scatter(x,y,color='g')
    plt.plot(x,model.predict(x),color='k')
    plt.show()

def plot_multi():
    
# calc_life_expectancy_one_feature()
calc_house_price_13features()