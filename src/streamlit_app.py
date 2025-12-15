import streamlit as st
from sklearn.datasets import load_breast_cancer # comes with builtin target/features
from gini import build_tree, classify, to_graphviz
from collections import Counter
import pandas as pd

def load_data():
    dataset = load_breast_cancer(as_frame=True) # as pandas bunch
    df = dataset['frame'] # convert to df
    labels = df['target'] # y
    features = df.drop(columns=['target']) # x
    label_count = Counter(labels) # -> Counter({0: 357, 1: 212})
    #print(df.columns)
    #print(df.head())
    #print(features.head())


features, labels, label_count = load_data()


'''https://docs.streamlit.io/develop/concepts/design/animate'''
'''
use grahpviz for the visualization bc its fast enough
'''
# TODO: create max_depth slilder


def main():
    tree = build_tree(labels, features, depth=0, maxdepth=5)
    st.title("Decision tree visualization")    
    st.header("Decision tree")

    # display tree
    dot = to_graphviz(tree)
    st.graphviz_chart(dot)

if __name__ == "__main__":
    main()
