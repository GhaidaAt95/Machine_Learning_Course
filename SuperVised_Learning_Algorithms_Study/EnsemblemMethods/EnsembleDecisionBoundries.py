'''
    This code built upon from : https://github.com/rasbt/mlxtend

    Ensembles smooth out decision boundries by averging out irregularities
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

def SmoothDecisionBoundries():
    '''
        Init 3 base classfiers:
            1. LogisticRegression
            2. RandomForestClassfier
            3. Support Vector Machin
        Init the ensemble classfier
    '''
    clf1 = LogisticRegression(random_state=0)
    clf2 = RandomForestClassifier(random_state=0)
    clf3 = SVC(random_state=0, probability=True)
    eclf = EnsembleVoteClassifier(clfs=[clf1,clf2,clf3], weights=[2,1,1], voting='soft')

    # Load Data
    X, y = iris_data()
    X = X[:,[0,2]]

    gs = gridspec.GridSpec(2,2)
    fig = plt.figure(figsize=(10,8))

    for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                             ['Logisitic Regression','Random Forest','RBF Kernel SVM','Ensemble'],
                             itertools.product([0,1], repeat=2)):
        clf.fit(X,y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X,y=y,clf=clf,legend=2)
        plt.title(lab)

    plt.show()

def plot_roc_curve(ytest, p_base_learners, p_ensemble, labels, ens_labels):
    plt.figure(figsize=(10, 8))
    plt.plot([0,1],[0,1], 'k--')
    print(p_base_learners.shape)
    cm =[plt.cm.rainbow(i)
        for i in np.linspace(0, 1.0, p_base_learners.shape[1]+1)]
    
    for i in range( p_base_learners.shape[1] ):
        p = p_base_learners[:,i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i+1])

    fpr, tpr, _ = roc_curve(ytest, p_ensemble)
    plt.plot(fpr, tpr, labels=ens_labels, c=cm[0])

    plt.xlabel('False Positive rate')
    plt.ylabel('True Positivie Rate')
    plt.title('ROC CURVE')
    plt.legend(frameon=False)
SmoothDecisionBoundries()