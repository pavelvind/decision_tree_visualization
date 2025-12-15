import streamlit as st
from sklearn.datasets import load_breast_cancer # comes with builtin target/features
from gini import build_tree, classify, to_graphviz
from collections import Counter
import pandas as pd

@st.cache_data
def load_data():
    dataset = load_breast_cancer(as_frame=True) # as pandas bunch
    df = dataset['frame'] # convert to df
    labels = df['target'] # y
    features = df.drop(columns=['target']) # x
    label_count = Counter(labels) # -> Counter({0: 357, 1: 212})
    #print(df.columns)
    #print(df.head())
    #print(features.head())
    return features, labels, label_count


@st.cache_data
def train_tree(features, labels, max_depth):
    return build_tree(features, labels, depth=0, maxdepth=max_depth)


#'''https://docs.streamlit.io/develop/concepts/design/animate'''
#'''use grahpviz for the visualization bc its fast enough'''
# TODO: create max_depth slilder

def main():
    
    st.title("Decision tree visualization") 
       
    features, labels, label_count = load_data()
    max_depth = st.slider("Max Depth", 1, 10, 5)
    with st.spinner("Building tree..."):
        tree = train_tree(features, labels, max_depth)

    # display tree
    dot = to_graphviz(tree)
    st.graphviz_chart(dot)

if __name__ == "__main__":
    main()
