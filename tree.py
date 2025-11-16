import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer # comes with builtin target/features
from collections import Counter
import json

dataset = load_breast_cancer(as_frame=True) # as pandas bunch
df = dataset['frame'] # convert to df
labels = df['target'] # y
features = df.drop(columns=['target']) # yx
label_count = Counter(labels) # -> Counter({0: 357, 1: 212})
#print(df.columns)
#print(df.head())
#print(features.head())

def giny(lbl):
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
    parent_impurity = giny(y)
    # loop thru all features to find the best giny
    for feature in x.columns:
        values = x[feature].unique()
        values.sort()
        
        # define threshold for each feature
        for threshold in values:
            x_left, y_left, x_right, y_right = split(x, y, feature, threshold)
            # if one side is empty continue
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            impurity_left = giny(y_left)
            impurity_right = giny(y_right)
            # weighted gain
            gain = parent_impurity - (impurity_left * (len(y_left)/len(y)) + impurity_right * (len(y_right)/len(y)))
            if gain > best_gain:
                best_gain = gain 
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain     

'''recursively build tree'''
def build_tree(x, y):
    
    best_feature, best_threshold, best_gain = find_best_split(x, y)
    
    # base case -> cannot be split further (gain == 0) 
    if best_gain == 0 or best_feature is None:
        counts = Counter(y)
        prediction = counts.most_common(1)[0][0]
        # return leaf
        return {"leaf": True, "prediction": prediction, "counts": counts}

    # recursive case
    x_left, y_left, x_right, y_right = split(x, y, best_feature, best_threshold)
    left_child = build_tree(x_left, y_left)
    right_child = build_tree(x_right, y_right)

    # this returns internal nodes that links children
    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_child,
        "right": right_child
    }

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

def main():
    tree = build_tree(features, labels)
    #print_tree(tree)
    sample = features.iloc[0]
    prediction = classify(sample, tree)
    print(f"sample actual={labels.iloc[0]} predicted={prediction}")

if __name__ == "__main__":
    main()
