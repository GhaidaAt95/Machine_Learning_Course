import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import learning_curve
import graphviz
import LearningCurves as LC
'''
    # Growing a tree involves deciding on :
        - Which features to use
        - What conditions to use for splitting
        - When to stop
    
    # Max_Depth: Lower  -> increase bias 
                 Higher -> increase variance
    # Min samples per split: minimum # of samples required to split a node
    
    # Max features: # of features to consider when looking for the best split.

    # min impurity split: Threshold for early stopping in tree growth. 
        - A node will split if its impurity is above the threshold
    
    # presort: whether to presort the data to speed up the finiding of the best splits in fitting

'''

iris_data = load_iris()

classification_tree = tree.DecisionTreeClassifier()

features = iris_data.data
target = iris_data.target

classification_tree.fit(features, target)

dot_data = tree.export_graphviz(classification_tree, out_file=None,
                                feature_names=iris_data.feature_names,
                                class_names=iris_data.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

print("length {}".format(len(target)))

plt.show()