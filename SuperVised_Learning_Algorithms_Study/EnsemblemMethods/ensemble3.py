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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import KFold
'''
    * Base learners take the orginal input anf generate a set of predictions
    * Original data set ordered as a matrix X of shape (n_samples, n_features)
    * The library of base learners output  new prediction as a new matrix Pbase of size(n_samples, n_base_learners)
    * The meta learner is trained on Pbase
    * Appropriate handling of training set
        - If we both train the base learners on X and let them predict X the meta learner will be training
                on the base earner's training error
    * We need a prediction matrix P that reflects test errors.
    
'''
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

# Define a library of base learners
base_learners = get_models()

# Define a meta learner
'''
    - linear models, kernerl-based(SVMs and KNNS), and decision tree based models
    - You can also use another ensemble as a meta learner 

    - Ex. here use Gradient Boostin Machine. 
        - Each of the 1000 decision trees to train on a random subset of 4 base learners and 50% of input data
        - This way GBM will be exposed to each base learner's strength in different neighborhoods of the input space
'''

meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    loss='exponential',
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005,
    random_state=SEED
)

# Define a procedure for generating train and test sets

# Blending

xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(xtrain, ytrain, test_size=0.5, random_state=SEED)

# Training set (Xtrain_base, ytrain_base) and Prediction Set (Xpred_base, ypred_base)

# Train Base Learners on a training set
def train_base_learners(base_learners, inp, out, verbose=True):
    if verbose: print("Fitting Models")
    for i, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..."% name, end=" ",flush=False)
        m.fit(inp, out)
        if verbose: print("Done")

train_base_learners(base_learners, xtrain_base, ytrain_base)

def predict_base_learners(pred_base_learners, inp, verbos=True):
    P = np.zeros((inp.shape[0], len(pred_base_learners)))

    if verbos: print("Generating Base Learner Predictions")
    
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbos: print("%s..."% name, end=" ",flush=False)
        p = m.predict_proba(inp)
        # We have 2 classes, need only predcitions for one class
        P[:,i] = p[:,1]
        if verbos: print("Done")
    
    return P

P_base = predict_base_learners(base_learners, xpred_base)
print("*"*70)
print("Predictions shape : {}".format(P_base.shape))
print("*"*70)
meta_learner.fit(P_base, ypred_base)

def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    P_pred = predict_base_learners(base_learners,inp,verbos=verbose)
    return P_pred, meta_learner.predict_proba(P_pred)[:,1]

P_pred, P = ensemble_predict(base_learners,meta_learner,xtest ,True)
print("Ensemble ROC-AUC score: %.3f"% roc_auc_score(ytest, P))
# score = 0.881 --> beats the best estimator from our previous benchmark 
# But it doesnt beat the simple average ensemble
# Because we trained the base learners and the meta_learner on only half of the data --> Info Lost
# Then use cross-Validation

def stacking(base_learners, meta_learner, X,y, generator):
    # Train final base learners for test time
    print("Fitting final base learners...",end="")
    #Final Base learners are trained on all data
    train_base_learners(base_learners,X,y,verbose=False)
    print("Done")

    # Generate predictoins for training meta learners
    #Outer loop:
    print("Generating Cross-validation predictions...")
    cv_preds, cv_y = [], []

    for i, (train_idx, test_idx) in enumerate(generator.split(X)):
        print('*'*70)
        print("split {}".format(i))
        print('*'*70)fo
        fold_xtrain, fold_ytrain = X[train_idx, :], y[train_idx]
        fold_xtest, fold_ytest = X[test_idx,:], y[test_idx]

        # Inner loop: Train the base learners on a training set and generate base learner predictions
        fold_base_learners = {name: clone(model)
                                for name, model in base_learners.items()}
        
        train_base_learners(fold_base_learners, fold_xtrain, fold_ytrain, verbose=False)

        fold_P_base = predict_base_learners(fold_base_learners, fold_xtest, verbos=False)

        cv_preds.append(fold_P_base)
        cv_y.append(fold_ytest)
        print("Fold %i done"%(i+1))

    print("CV-predictions done")

    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)

    #Train meta learner
    print("Fitting meta learner...",end="")
    meta_learner.fit(cv_preds, cv_y)
    print("Done")

    return base_learners, meta_learner

# clone: constructs a new estimator with the same parameters
#        It yields a new estimator with the same parameters that has not been fit on any data
cv_base_learners, cv_meta_learner = stacking(
    get_models(), clone(meta_learner), xtrain.values, ytrain.values, KFold(n_splits=2))

'''
    * Kfold provides train/test indices to split data in train/test sets
    * split dataset into k consecutive folds (without shuffling is the default)

'''

P_pred, p = ensemble_predict(cv_base_learners, cv_meta_learner, xtest, verbose=False)
print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(ytest, p))





