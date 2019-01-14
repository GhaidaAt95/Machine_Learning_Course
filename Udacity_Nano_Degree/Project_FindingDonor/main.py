import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import visuals as vs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
'''
    # Goal: Construct a model that accurately predicts whether an individual makes more than $50,000
'''

def print_section(printthis="",title=None):
    print('*'*100)
    if title != None:
        print("\t\t\t",title)
    print(printthis)
    print('*'*100)
# Data : Income daa collected from the 1994 U.S Census
data = pd.read_csv('census.csv')

print_section(data.head())

################# Data Exploration #################

# Total number of records
n_records = data.shape[0]
print(n_records)

# Number of individauls making more than 50,000 annually
n_greater_50k = data.where(data['income']==">50K").count()[0]

# Number of individauls making at most 50,000 annually
n_at_most_50k = data.where(data['income']=="<=50K").count()[0]

# Percenteger of individauls making more than 50,000 annually 
greater_percent = n_greater_50k / (n_greater_50k+n_at_most_50k)

Info = '''Total NUmber of records: {}
Individuals making more than $50,000: {}
Individuals making at most $50,000: {}
Percentage of individuals making more than $50,000: {}'''.format(n_records,n_greater_50k,n_at_most_50k,greater_percent)
print_section(Info)

'''
    Features :
        1) Continous:
            1. Age
            2. education-num
            3. capital-gain
            4. capital-loss
            5. hours-per-week
        2) Discrete/Cateogrical Features:
            1. workclass : Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
            2. education : Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th,Doctrate, 5th-6th, Preschool
            3. martial-status : Married-civ-spouse, Divorced, Never-married, Seperated, Widowed, Married-spouse-abscent, Married-AF-spouse
            4. occupation : Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, 
                            Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
            5. relationship : Wife, Own-child, Husband, Not0in-family, Other-relative, Unmarried
            6. race : Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
            7. sex : female, male
            8. native-country : 
'''

################# Preparing / Preprocessing Data #################

income_raw = data['income']
features_raw = data.drop('income',axis=1)

### Transform Skewed Continous Features ###
'''
    A fetures values may tend to lie near a single #, but will also
    have a non-trivial number of vastly larger or smaller values than
    that single # --> Those can affect a senstive algorithm --> 
    The range need to be normalized
'''
# Visualize the skewed continuous features
vs.distribution(data,features=['capital-gain','capital-loss'])

### Logarithmic transformation on the data ###
# Thus, very large and very small values do not negatively affect the performance of a learning algorithm

skewed = ['capital-gain','capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
# x + 1 since log(0) is not defined and thus we tanslate the values by a small amount above 0
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x+1))

# Visualize the transformed data features
vs.distribution(features_log_transformed,features=['capital-gain','capital-loss'],transformed=True)

### Normalizing Numerical Features ###

'''
    Normalizing ensures that each feature is treated equally when applying supervised learners

'''
# Transforms features by scaling each feature to a given range.
scaler = MinMaxScaler()
numerical = ['age','education-num','capital-gain','capital-loss','hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
print_section(features_log_minmax_transform.head(n=5),title="After Normalizing")

### One-hot enconding of categorical variables ###
features_final = pd.get_dummies(features_log_minmax_transform)

# Convert income from "<=50K" to 0 and ">50K" to 1
income = income_raw.apply(lambda x : 0 if x == '<=50K' else 1)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print(encoded)

### Shuffle and Split Data ###
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)
 
Info = '''          After shuffling and splitting
Training set has {} samples.
Testing set has {} samples.
'''.format(X_train.shape[0],X_test.shape[0])

print_section(Info)

################# Evaluatin Model Performance #################
'''
    Investigate four different algorithms
        1)
        2)
        3)
        4) Naive Predictor
    Choose from:
        - Gaussian Naive Bayes 
        - Ensemble Mehods(Bagging, AdaBoost, RandomForest, Gradient Boosting)
        - K-Nearest Neighbors (KNeighbors)
        - Stochastic Gradient Descent Classifier (SGDC)
        - Support Vector Machines (SVM)
        - Logistic Regression
'''

### Naive Predictor ( All individuals made more than $50,000 )###
'''
    - Generally : Most individuals do not makee more than $50,000 <-- [Naive Statement]
    - Accuracy  : (#predicted correctly)/(#of test points)
    - Precision : Out of all emails marked Spam, how many are actually spam
                  (T+ve)/(T+ve + F+ve)
    - Recall    : Out of all Sick patients, how many were identified
                  (T+ve)/(T+ve + F-ve)
    - F_Score   : Weighted average of the precision and recall scores
    - Our model cares more about precision than recall. 
        Out of all marked '>50K', how many are actually making >50K
    
    - Generating a naive predictor to show what a base model without intelligience would look like.
        - In real world a base model would be either: 
            - Results of a previous model
            OR
            - Based on a research paper upon which you are looking to improve

    - When there is no benchmark --> Getting a result better than random choice is a place to start from
'''

true_positives = np.sum(income)
false_positives = income.count() - true_positives
true_negatives = 0
false_negatives = 0

accuracy = true_positives / (income.count())
recall = true_positives / ( true_positives + false_negatives )
precision = true_positives / (true_positives + false_positives)

# f0.5_score = 1.25 ( precision * recall / ((0.25presision) + recall ))
f_score =  (1 + (0.5**2))*( ( precision * recall ) / ( ((0.5**2) * precision ) + recall ))

Info = '''Naive Predictor: [ Accuracy Score: {:.4f} , F-score: {:.4f} ]
'''.format(accuracy,f_score)
print_section(Info)

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
        inputs: 
            - learner: Learning Algorithm to be trained and predicted on
            - Sample_size: The size of samples to be drawn from the training set
    '''
    results = {}

    start = time()
    learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time()

    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test[:sample_size])
    predictions_train = learner.predict(X_train[:sample_size])
    end = time()

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:sample_size],predictions_train)
    
    results['acc_test'] = accuracy_score(y_test[:sample_size],predictions_test)

    results['f_train'] = fbeta_score(y_train[:sample_size],predictions_train,beta=0.5)
        
    results['f_test'] =  fbeta_score(y_test[:sample_size],predictions_test,beta=0.5)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results

clf_A = RandomForestClassifier(n_estimators=10)
clf_B = GaussianNB()
clf_C = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),n_estimators=4)

samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.001)
print("1) {} 2) {} 3) {}".format(samples_100,samples_10,samples_1))
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, f_score)
plt.show()