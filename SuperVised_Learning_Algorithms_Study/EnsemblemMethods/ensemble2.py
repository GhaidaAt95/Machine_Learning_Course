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
from sklearn.metrics import roc_curve

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
P2 = train_predict2(models)
p3 = P.apply(lambda pred: 1 * (pred>=0.5) - ytest.values)

# writer = pd.ExcelWriter('output.xlsx')
# P.to_excel(writer, 'sheet1')
# P2.to_excel(writer, 'sheet2')
# p3.to_excel(writer, 'sheet3')
# writer.save()

from mlens.visualization import corrmat
import seaborn as sns
sns.set()
sns.diverging_palette(240, 10, n=9)
corrmat(P.corr(), inflate=False)
'''
    Errors are significantly correlated --> Expected for models that perform well

    Outliers are hard to get right

    BUT most correlations are in the 50%-80% span, there is a decent room for imporovement
'''
## Error correlations on a class prediction basis
corrmat(P.apply(lambda pred: 1*(pred >= 0.5) - ytest.values).corr(), inflate=False)

#### Ensemble: Average predictions
print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))

### Ensembes power as an error correction mechanism --> It smootha out descision boundries by Averaging out irregulaarities.
# import EnsembleDecisionBoundries as ES
# ES.SmoothDecisionBoundries()


'''
    - A non-linear meta learer is able for each training point to adjust which model it relies on
        - Some models for example have precision but sacrificing recall 
'''

def plot_roc_curve(ytest, p_base_learners, p_ensemble, labels, ens_label):
    plt.figure(figsize=(10, 8))
    plt.plot([0,1],[0,1], 'k--')
    cm =[plt.cm.rainbow(i)
        for i in np.linspace(0, 1.0, p_base_learners.shape[1]+1)]
    
    for i in range( p_base_learners.shape[1] ):
        p = p_base_learners.iloc[:,i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i+1])

    fpr, tpr, _ = roc_curve(ytest, p_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False Positive rate')
    plt.ylabel('True Positivie Rate')
    plt.title('ROC CURVE')
    plt.legend(frameon=False)

plot_roc_curve(ytest=ytest,p_base_learners= P,p_ensemble=P.mean(axis=1),labels=['svm','knn','nb','nn','rf','gb','lr'],ens_label='Avg Ensemble')

'''
    - Some models perform worse than others and their impact on the prediction average can be huge

    - In this example all models underrepesent republican donations But some models are worse than others
'''

P5 = P.apply(lambda x: 1*(x>=0.5).value_counts(normalize=True))
P5.index = ["DEM","REP"]
plt.figure(figsize=(10, 8))
P5.loc["REP",:].sort_values().plot(kind="bar")
plt.axhline(0.25, color='k',linewidth=0.5)
plt.text=(0,0.23,"True share republicans")

## Try remove the worst MLP and check the performnce
include = [c for c in P.columns if c not in ['mlp-nn']]
print("Truncated ensemble ROC-AUC Score: %.3f"% roc_auc_score(ytest, P.loc[:,include].mean(axis=1)))


plt.show()