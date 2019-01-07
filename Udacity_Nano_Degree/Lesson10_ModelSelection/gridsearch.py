import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import random
# Training and Testing Sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
# Metrics
from sklearn.metrics import f1_score, make_scorer

from load_pts import load_pts
from plot_model import plot_model

X, y = load_pts('data.csv')
plt.show()

#Fixing a random seed
random.seed(42)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Define the model(with default hyper-parametes)
clf = DecisionTreeClassifier(random_state=42)

# fit the model
clf.fit(X_train, y_train)

#make predictions
train_predictions = clf.predict(X_train)
test_predictions  = clf.predict(X_test)

plot_model(X,y,clf)
print('The Training F1 Score is', f1_score(train_predictions, y_train))
print('The Testing F1 Score is', f1_score(test_predictions, y_test))

clf2 = DecisionTreeClassifier(random_state=42)

# Create the parameters list {max_depth, min_samples_leaf, and min_samples_split}
parameters = {'max_depth':[2,4,6,8,10],'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]}

# Make an fbeta_score scoring object
scorer = make_scorer(f1_score)

# perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf2, parameters, scoring=scorer,cv=3)

# Fir the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Fit the new model
best_clf.fit(X_train,y_train)

# Make predictions using the new model
best_train_predictions = best_clf.predict(X_train)
best_test_predictions = best_clf.predict(X_test)

# Calculate the f1_score of the new model
print("*******************************")
print('The New Training F1 Score is',f1_score(best_train_predictions, y_train))
print('The NewTesting F1 Score is',f1_score(best_test_predictions, y_test))

plot_model(X,y,best_clf)

print(best_clf)


# parameters = {'kernel':['ploy', 'rbf'], 'C':[0.1,1,10]}
# scorer = make_scorer(f1_score)
# # Create a GridSearch object --> Use this object to fit the data
# grid_obj = 	GridSearchCV(clf, parameters, scoring=scorer)
# #fit the data
# grid_fit = grid_obj.fit(X, y)
# best_clf = grid_fit.best_estimator_
