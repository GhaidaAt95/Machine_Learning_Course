import numpy as np
import pandas as pd
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
from mlens.ensemble import SuperLearner
from sklearn.metrics import roc_curve


def get_train_test(test_size=0.95):
    # Split Data into train and test sets
    # create a target pd.series in which 1 = REP and 0 = DEM
    y = 1 * (df.cand_pty_affiliation == "REP")
    x = df.drop(["cand_pty_affiliation"], axis=1)
    x = pd.get_dummies(x, sparse=True)
    x.drop(x.columns[x.std() == 0], axis=1, inplace=True)
    return(train_test_split(x, y, test_size=test_size, random_state=SEED))

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

if __name__ == '__main__':
    SEED = 222
    np.random.seed(SEED)

    df = pd.read_csv('input.csv')

    xtrain, xtest, ytrain, ytest = get_train_test()

    # Define a library of base learners
    base_learners = get_models()

    # Define a meta learner
    meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    loss='exponential',
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005,
    random_state=SEED
    )


    # Init the ensmble, here do 10-Folds
    s1 = SuperLearner(
        folds=10,
        random_state=SEED,
        verbose=2, 
        backend='multiprocessing'
    )

    # Add base learners and the meta learner
    s1.add(list(base_learners.values()), proba=True)
    s1.add_meta(meta_learner, proba=True)

    s1.fit(xtrain, ytrain)
    p_s1 = s1.predict(xtest)

    print("\nSuper Learner ROC-AUC score : %.3f"% roc_auc_score(ytest, p_s1[:,1]))
    