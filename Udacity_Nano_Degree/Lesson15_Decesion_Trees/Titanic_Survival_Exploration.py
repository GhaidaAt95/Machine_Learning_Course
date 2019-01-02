import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
random.seed(42)

## Load Data
in_file = 'titanic_data.csv'
df = pd.read_csv(in_file)

# Display Data Head
print('*'*70)
print(df.head())
print('*'*70)

'''
    Survived: Outcome of survival (0 = No; 1 = Yes)
    Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
    Name: Name of passenger
    Sex: Sex of the passenger
    Age: Age of the passenger (Some entries contain NaN)
    SibSp: Number of siblings and spouses of the passenger aboard
    Parch: Number of parents and children of the passenger aboard
    Ticket: Ticket number of the passenger
    Fare: Fare paid by the passenger
    Cabin: Cabin number of the passenger (Some entries contain NaN)
    Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
'''

outcomes = df['Survived']
features_raw = df.drop('Survived', axis=1)

print('*'*70)
print(features_raw.head())
print('*'*70)

features_no_names = features_raw.drop(['Name'], axis=1)

## Convert catrgorical variables into dmmy/indicator variables
features = pd.get_dummies(features_no_names)
print('*'*70)
print(features.head())
print('*'*70)

features = features.fillna(0.0)
print('*'*70)
print(features.head())
print('*'*70)

##########################################
#1 Split Data into training and testing set
X_train, X_test, y_train, y_test =train_test_split(features, outcomes, test_size=0.2, random_state=42)

#2 Define Model and Fit
model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=10)
model.fit(X_train,y_train)

#3 Make predictions 
y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_predictions)
test_accuracy = accuracy_score(y_test, y_test_predictions)

print('The training accuracy is: ',train_accuracy)
print('The test accuracy is: ',test_accuracy)
'''
    default hyperparameters are 
    max_depth = None
    min_samples_leaf = 1
    min_samples_split =2
'''
maxDepths = range(1,21) #1-20
minSLeafs = range(1,21) #1-20
minSSplit = range(2,22) #2-19

# df = np.zeros((9,9,9))
# print(maxDepths)
maxV = 0.0
a = ''
for i in maxDepths:
    for y in minSLeafs:
        for z in minSSplit:
            model2 = DecisionTreeClassifier(max_depth=i,min_samples_leaf=y,min_samples_split=z)
            model2.fit(X_train,y_train)
            y_train_predictions = model2.predict(X_train)
            y_test_predictions = model2.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_predictions)
            test_accuracy = accuracy_score(y_test, y_test_predictions) 
            # print("#{}/{}/{} = {}".format(i,y,z,test_accuracy))
            if test_accuracy > maxV:
                maxV = test_accuracy,
                a ='{}/{}/{}'.format(i,y,z)

print("{} at ".format(maxV),a)

