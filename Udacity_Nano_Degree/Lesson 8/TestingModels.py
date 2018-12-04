#Import Statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = np.asarray(pd.read_csv('class_data_2.csv',header=None))
#data2 = pd.read_csv('class_data.csv',header=0)
#print(data.head())
#print(data2.head())

X = data[:,0:2]
y = data[:,2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

model = None

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test,y_pred)

print('accuracy_score = : ',acc)
0.54