'''
    Source Study: https://www.dataquest.io/blog/introduction-to-ensembles/
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from print_graph import print_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

print(df.head())

df.cand_pty_affiliation.value_counts(normalize=True).plot(kind="bar", title="Share of No. donations")

'''
    Features about:
        1) Donor:
            1. entity_tp : Individual or Organization
            2. state
            3. classification: scientfic field they work in
        2) Transaction:
            1. rtp_tp: Identifies contributions made during campaigns
            2. transaction_tp: form of contribution made
            3. cycle: year
            4. transaction_amt
        3) Recipient:
[Traget]    1. cand_pty_affiliation
            2. cand_office_st
            3. cand_office
            4. cand_status
'''
t1 = DecisionTreeClassifier(max_depth=1, random_state=SEED)
t1.fit(xtrain, ytrain)
p1 = t1.predict_proba(xtest)[:,1]

print("Decision Tree ROC_AUS SCORE: %.3f"%roc_auc_score(ytest,p1))
print_graph(t1, xtrain.columns,"MaxDepth1")

'''
    - With a depth of 1, the ROC is 0.672 which is sligghtly better than a random 
    - Its not making use of the data we have
'''

t2 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
t2.fit(xtrain,ytrain)
p2 = t2.predict_proba(xtest)[:,1]

print("Descision tree ROC_AUC score: %.3f"%roc_auc_score(ytest,p2))
print_graph(t2,xtrain.columns,"MaxDepth3")

'''
    - This model is not much better than the simple decision tree
    - Making it deeper will just causes it too overfit

    - Creatinh several decision trees and combine them
        - How can we force the decision tree to investigate other patterns than
            those in the 2 models above ???
            - Simple solution: remove features that appear early in the tree
                Ex. transaction_amt
'''

drop = ["transaction_amt"]

xtrain_slim = xtrain.drop(drop, axis=1)
xtest_slim = xtest.drop(drop, axis=1)

t3 = DecisionTreeClassifier(max_depth=3, random_state=SEED)
t3.fit(xtrain_slim, ytrain)
p3 = t3.predict_proba(xtest_slim)[:,1]

print("Descision tree ROC_AUC score: %.3f"%roc_auc_score(ytest,p3))
print_graph(t3,xtest_slim.columns,"MaxDepth3AndSlim")

'''
    - ROC is similar But the share of Republican donation increased to 7.3%
    - In contrast, the 1st tree focused most of the rules related to the transaction itself
        But here more focused on the residency of the candidate

    - Two models that by themselves have similar predictive power, But
        operate on different rules.

    - Make different prediction errors, which can be averged out with an ensemble
'''

### Check Error Correlation
    # Highly correlated Errors -> Poor Ensembles

Correlated = pd.DataFrame({"full_data": p2,
              "red_data": p3}).corr()
print(Correlated)

'''
    There is some correlation

    There is still a good deal or prediction variance to exploit

    To build our 1st ensemble --> average the 2 models predictions
'''

pred = np.mean([p2, p3], axis=0)

print("Average of decision tree ROC-AUC scroe: %.3f"%roc_auc_score(ytest,pred))

'''
    - The score increased
    - Maybe if we have more diverse trees the core would be higher
    - How should we choose which features to exclude when designing the descion trees?

    WAYS: 
        1) Bootstrap Aggregating (Bootstrap): 
            Randomly select a subset of features -> fit one decision tree
                on each draw and average their predictions
            - RandomForest: 
                implements a level of differentioation because each tree will split based on different feature
'''
rf = RandomForestClassifier(
    n_estimators=10,
    max_features=3,
    random_state=SEED
)

rf.fit(xtrain, ytrain)
p4 = rf.predict_proba(xtest)[:,1]
print("Average of decision tree ROC-AUC score: %.3f" % roc_auc_score(ytest, p4))


plt.show()
