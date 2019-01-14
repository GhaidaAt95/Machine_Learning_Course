import graphviz
from sklearn.tree import export_graphviz

def print_graph(clf, feature_names, name_output):
    graph = export_graphviz(
        clf,
        label="root",
        proportion=True,
        impurity=False,
        out_file= None,
        feature_names=feature_names,
        class_names={0:'D', 1:'R'},
        filled = True,
        rounded=True
    )
    graph = graphviz.Source(graph)
    graph.render(name_output)
