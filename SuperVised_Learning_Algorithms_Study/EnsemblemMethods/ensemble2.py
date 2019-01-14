'''
    Source Study: https://www.dataquest.io/blog/introduction-to-ensembles/
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 222
np.random.seed(SEED)

df = pd.read_csv('input.csv')

def get_train_test(test_size=0.95):
    # Split Data into train and test sets
    # create a target pd.series in which 1 = REP and 0 = DEM
    y = 1 * (df.cand_pty_affiliation == "REP")
    x = df.drop(["cand_pty_affiliation"], axis=1)
    x = pd.get_dummies(x, sparse=True)
    x.drop(x.columns[x.std() == 0], axis=1, inplace=True)
    return(train_test_split(x, y, test_size=test_size, random_state=SEED))

xtrain, xtest, ytrain, ytest = get_train_test()

def get_models():
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models

def train_predict(model_list):
    predictions = np.zeros((ytest.shape[0], len(model_list)))
    predictions = pd.DataFrame( predictions )

    print("Fitting Models")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..."% name, end=" ",flush=False)
        m.fit(xtrain, ytrain)
        predictions.iloc[:,i] = m.predict_proba(xtest)[:,1]
        cols.append(name)
        print("done")

    predictions.columns = cols
    print("Done\n")
    return predictions

def train_predict2(model_list):
    predictions = np.zeros((ytest.shape[0], len(model_list)))
    predictions = pd.DataFrame( predictions )

    print("Fitting Models")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..."% name, end=" ",flush=False)
        m.fit(xtrain, ytrain)
        predictions.iloc[:,i] = m.predict(xtest)
        cols.append(name)
        print("done")

    predictions.columns = cols
    print("Done\n")
    return predictions

def score_models(P, y):
    print("Scoring models")
    for m in P.columns:
        score = roc_auc_score(ytest, P.loc[:,m])
        print("%-26s: %.3f"%(m, score))
    print("Done\n")

models = get_models()
P = train_predict(models)
score_models(P, ytest)
# P2 = train_predict2(models)

writer = pd.ExcelWriter('output.xlsx')
P.to_excel(writer, 'sheet1')
# P2.to_excel(writer, 'sheet2')
writer.save()

from mlens.visualization import corrmat
import seaborn as sns; 
sns.set()
sns.diverging_palette(240, 10, n=9)
corrmat(P.corr(), inflate=False)

## Error correlations on a class prediction basis
corrmat(P.apply(lambda pred: 1*(pred >= 0.5) - ytest.values).corr(), inflate=False)

print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))

plt.show()