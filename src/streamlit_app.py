import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes # comes with builtin target/features
from gini import build_tree, classify, to_graphviz, highligh_gprah
from collections import Counter
import pandas as pd

@st.cache_data
def load_data(dataset_name, _dataset_loader):
    dataset = _dataset_loader(as_frame=True) # as pandas bunch
    df = dataset['frame'] # convert to df
    labels = df['target'] # y
    features = df.drop(columns=['target']) # x
    label_count = Counter(labels) # -> Counter({0: 357, 1: 212})
    #print(df.columns)
    #print(df.head())
    #print(features.head())
    target_names = getattr(dataset, 'target_names', None)
    return features, labels, label_count, target_names


@st.cache_data
def train_tree(features, labels, max_depth):
    return build_tree(features, labels, depth=0, maxdepth=max_depth)


#'''https://docs.streamlit.io/develop/concepts/design/animate'''
#'''use grahpviz for the visualization bc its fast enough'''

def main():
    
    st.title("Decision tree visualization") 

    datasets = {"Breast Cancer": load_breast_cancer,
                "Iris": load_iris,
                "Wine": load_wine,
                "Diabetes": load_diabetes}
    
    dataset_name = st.selectbox("Choose dataset", list(datasets.keys()))
    features, labels, label_count, target_names = load_data(dataset_name, datasets[dataset_name])

    # explain label
    if target_names is not None:
        st.write("Class Labels:", {i: name for i, name in enumerate(target_names)})
    else:
        st.text(str(labels.head()))
    
    max_depth = st.slider("Max Depth", 1, 10, 5)
    with st.spinner("Building tree..."):
        tree = train_tree(features, labels, max_depth)

    # display tree
    dot = to_graphviz(tree)
    st.graphviz_chart(dot)

    # ask for user sample
    sample = pd.DataFrame([features.mean()], columns=features.columns)
    st.text("Sample with mean values")
    sample = st.data_editor(sample)
    if st.button("classify"):
        x = sample.iloc[0]
        prediction = classify(x, tree)
        if target_names is not None:
            st.success(f"Prediction: {target_names[prediction]}")
        else:
            st.success(f"Prediction: {prediction}")
        dot = highligh_gprah(tree, x)
        st.graphviz_chart(dot)
        
    
if __name__ == "__main__":
    main()
