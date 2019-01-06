import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_File = '.\\smsspamcollection\\SMSSpamCollection'

df = pd.read_table(data_File, header=None, names=['Classification','Messages'])
print('*'*70,"\n Data")
print(df.head())

## 1: Spam , 0: Ham
df['Classification'] = df.Classification.map({'spam':1, 'ham':0})
df = df.rename(columns={'Classification':'Spam'})
print('\n','*'*70,"\n Data with categorical values")
print(df.head())

'''
    CountVectorizer:
        - Tokenizes the string and gives an INT ID to each token
        - Counts the occurence of each of those tokens
        - Converts all tokenized words to thier lower case form
            - lowercase parameter
        - Ignores all puncuation "hello!" == "hello"
            - token_pattern parameter
        - *** Stop words paramter if set to english it will ignore some of
            the most commonly used words such as 'am','an','and','the',...
'''


X_train, X_test, y_train, y_test = train_test_split(df['Messages'],df['Spam'],random_state=1)

print("Number of rows in the total set: {}".format(df.shape[0]))
print("Number of rows in the training set: {}".format(X_train.shape[0]))
print("Number of rows in the testing set: {}".format(X_test.shape[0]))

count_vector = CountVectorizer()

# Fit the training data and then return the DTM
training_data = count_vector.fit_transform(X_train)

# Tranform testing data and return its corresoinding DTM
testing_data = count_vector.transform(X_test)

# Multinominal Niave Bayes - for classification of discrete features
naive_bayes = MultinomialNB()
naive_bayes.fit(X=training_data, y=y_train)

predictions = naive_bayes.predict(X=testing_data)

# Accuracy : (T+ve + T-ve) / Total # od data
print('Accuracy Score : ', format(accuracy_score(y_test, predictions)))
# Precision : from all that is classified spam, how many are actually spam
# p = T+ve / (T+ve + F+ve)
print('Precision Score : ', format(precision_score(y_test, predictions)))
# Recall: from all spam emails, how many were identified
# R = T+ve / (T+ve + F-ve)
print('Recall Score : ', format(recall_score(y_test, predictions)))
# F1_Score = Harmonic Mean, and it give equal importance to 
# both precision and Recall. It will be closer to the smallest
# between precision and Recall. 
# F1_score = (2*Precision*Recall)/(Precision + Recall)
print('F1 Score : ',format(f1_score(y_test, predictions)))