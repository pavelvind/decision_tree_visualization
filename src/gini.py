import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer # comes with builtin target/features
from collections import Counter
import graphviz

dataset = load_breast_cancer(as_frame=True) # as pandas bunch
df = dataset['frame'] # convert to df
labels = df['target'] # y
features = df.drop(columns=['target']) # yx
label_count = Counter(labels) # -> Counter({0: 357, 1: 212})
#print(df.columns)
#print(df.head())
#print(features.head())

def gini(lbl):
    impurity = 1
    counts = Counter(lbl)
    for label, count in counts.items(): # returns pairs -> label, count
        probability = count/len(lbl)
        impurity -= probability**2
    return impurity

def split(x, y, feature, threshold):
    left = x[feature] < threshold
    right = ~ left

    return (
        x[left],
        y[left],
        x[right],
        y[right]
    )


def find_best_split(x, y):
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    parent_impurity = gini(y)
    # loop thru all features to find the best gini
    for feature in x.columns:
        values = x[feature].unique()

        # PERFROMANCE: throw away the numbers and create 10 representatives
        if len(values) > 10:
            values = np.percentile(values, np.linspace(0, 100, 10))
        else:
            values.sort()
        
        # define threshold for each feature
        for threshold in values:
            x_left, y_left, x_right, y_right = split(x, y, feature, threshold)
            # if one side is empty continue
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            impurity_left = gini(y_left)
            impurity_right = gini(y_right)
            # weighted gain
            gain = parent_impurity - (impurity_left * (len(y_left)/len(y)) + impurity_right * (len(y_right)/len(y)))
            if gain > best_gain:
                best_gain = gain 
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain     

'''recursively build tree'''
def build_tree(x, y, depth=0, maxdepth=5):
    
    best_feature, best_threshold, best_gain = find_best_split(x, y)

    # base case -> depth limit reached
    # base case -> cannot be split further (gain == 0) 
    if depth > maxdepth or best_gain == 0 or best_feature is None:
        counts = Counter(y)
        prediction = counts.most_common(1)[0][0]
        return {"leaf": True, "prediction": prediction, "counts": counts}
    

    # recursive case
    x_left, y_left, x_right, y_right = split(x, y, best_feature, best_threshold)
    left_child = build_tree(x_left, y_left, depth+1, maxdepth)
    right_child = build_tree(x_right, y_right, depth+1, maxdepth)

    # this returns internal nodes that links children -> multiple dicitonaries
    # TODO: need to export this to DOT langugage for graphviz (helper function)
    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_child,
        "right": right_child
    }

'''this function converts the dictionary into dot format language for graphviz'''
def _graph():
    dot = graphviz.Digraph('DecisionTree', comment='Decision Tree')
    dot.attr(
        bgcolor='transparent',
        pad='0.25',
        nodesep='0.3',
        ranksep='0.55',
        splines='ortho'
    )
    dot.attr(
        'node',
        shape='box',
        style='rounded,filled',
        fontname='Helvetica',
        fontsize='11',
        color='#d9e2dc',
        fontcolor='#17211b',
        penwidth='1.2',
        margin='0.18,0.11'
    )
    dot.attr(
        'edge',
        fontname='Helvetica',
        fontsize='9',
        color='#b9c5bd',
        fontcolor='#66736b',
        arrowsize='0.65'
    )
    return dot


def _leaf_label(node):
    counts = '  ·  '.join(f'{label}: {count}' for label, count in node['counts'].items())
    samples = sum(node['counts'].values())
    return f"Prediction · class {node['prediction']}\n{samples} samples  |  {counts}"


def _leaf_color(prediction):
    colors = ['#e4f3ed', '#fff0d6', '#eee9fb', '#fbe8ec']
    return colors[int(prediction) % len(colors)]


def to_graphviz(tree):
    dot = _graph()

    def add_nodes_edges(node, parent_id=None, edge_label=""):
        node_id = str(id(node))

        if node['leaf']:
            dot.node(
                node_id,
                label=_leaf_label(node),
                fillcolor=_leaf_color(node['prediction'])
            )
        else:
            label = f"{node['feature']}\n< {node['threshold']:.2f}"
            dot.node(node_id, label=label, fillcolor='#f6f8f6')
            add_nodes_edges(node['left'], parent_id=node_id, edge_label="True")
            add_nodes_edges(node['right'], parent_id=node_id, edge_label="False")

        if parent_id:
            dot.edge(parent_id, node_id, label=edge_label)

    add_nodes_edges(tree)
    return dot


def highlight_graph(tree, sample):
    dot = _graph()

    def add_nodes_edges(node, parent_id=None, edge_label="", is_on_path=True):
        node_id = str(id(node))
        path_color = '#16805d'

        if node['leaf']:
            dot.node(
                node_id,
                label=_leaf_label(node),
                fillcolor='#daf2e7' if is_on_path else _leaf_color(node['prediction']),
                color=path_color if is_on_path else '#d9e2dc',
                penwidth='2.2' if is_on_path else '1.2'
            )
        else:
            label = f"{node['feature']}\n< {node['threshold']:.2f}"
            dot.node(
                node_id,
                label=label,
                fillcolor='#e4f3ed' if is_on_path else '#f6f8f6',
                color=path_color if is_on_path else '#d9e2dc',
                penwidth='2.2' if is_on_path else '1.2'
            )

            goes_left = is_on_path and sample[node['feature']] < node['threshold']
            goes_right = is_on_path and not goes_left
            add_nodes_edges(
                node['left'],
                parent_id=node_id,
                edge_label="True",
                is_on_path=goes_left
            )
            add_nodes_edges(
                node['right'],
                parent_id=node_id,
                edge_label="False",
                is_on_path=goes_right
            )

        if parent_id:
            dot.edge(
                parent_id,
                node_id,
                label=edge_label,
                color=path_color if is_on_path else '#b9c5bd',
                fontcolor=path_color if is_on_path else '#66736b',
                penwidth='2.2' if is_on_path else '1.0'
            )

    add_nodes_edges(tree)
    return dot


# Backwards-compatible alias for the original public API.
highligh_gprah = highlight_graph
            
           

def classify(sample, node):
    # base case -> leaf reached
    if node['leaf']:
        return node['prediction']
    # compare values @ the current node -> threshold (defined for each value)
    if sample[node['feature']] < node['threshold']:
        return classify(sample, node['left'])
    else: 
        return classify(sample, node['right'])
    

def print_tree(node):
    if node['leaf']:
        print(node)
        return
    print(node)
    print_tree(node['left'])
    print_tree(node['right'])
