from collections import Counter
from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

# Streamlit Cloud can launch this file with either the repository root or `src`
# as its working directory. Resolve sibling modules from this file's location.
SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from gini import build_tree, classify, highlight_graph, to_graphviz


st.set_page_config(
    page_title="Decision Tree Visualizer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATASETS = {
    "Breast Cancer": load_breast_cancer,
    "Iris": load_iris,
    "Wine": load_wine,
}

DATASET_DESCRIPTIONS = {
    "Breast Cancer": "Diagnostic measurements from breast mass images.",
    "Iris": "Flower measurements across three iris species.",
    "Wine": "Chemical analysis of wines from three cultivars.",
}


@st.cache_data
def load_data(dataset_name, _dataset_loader):
    dataset = _dataset_loader(as_frame=True)
    df = dataset["frame"]
    labels = df["target"]
    features = df.drop(columns=["target"])
    label_count = Counter(labels)
    target_names = getattr(dataset, "target_names", None)
    return features, labels, label_count, target_names


@st.cache_data
def train_tree(features, labels, max_depth):
    return build_tree(features, labels, depth=0, maxdepth=max_depth)


def class_legend(target_names):
    if target_names is None:
        return ""

    chips = "".join(
        f'<span class="class-chip"><b>{index}</b>{escape(str(name))}</span>'
        for index, name in enumerate(target_names)
    )
    return f'<div class="class-legend">{chips}</div>'


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --ink: #16201a;
                --muted: #657069;
                --line: #dfe6e1;
                --surface: #ffffff;
                --canvas: #f5f7f4;
                --accent: #16805d;
                --accent-soft: #e5f3ed;
            }

            .stApp {
                background: var(--canvas);
                color: var(--ink);
            }

            .block-container {
                max-width: 1240px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }

            [data-testid="stSidebar"] {
                background: #eef2ee;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #5f6b64;
            }

            .sidebar-brand {
                display: flex;
                align-items: center;
                gap: .7rem;
                margin: .2rem 0 2rem;
                color: var(--ink);
                font-size: 1.05rem;
                font-weight: 720;
                letter-spacing: -.02em;
            }

            .brand-mark {
                display: grid;
                place-items: center;
                width: 2rem;
                height: 2rem;
                border-radius: .65rem;
                background: var(--accent);
                color: white;
                font-size: 1rem;
                box-shadow: 0 7px 18px rgba(22, 128, 93, .18);
            }

            .hero {
                padding: 1.1rem 0 2rem;
                border-bottom: 1px solid var(--line);
                margin-bottom: 1.7rem;
            }

            .eyebrow {
                display: flex;
                align-items: center;
                gap: .45rem;
                color: var(--accent);
                font-size: .75rem;
                font-weight: 750;
                letter-spacing: .13em;
                text-transform: uppercase;
            }

            .eyebrow-dot {
                width: .45rem;
                height: .45rem;
                border-radius: 50%;
                background: var(--accent);
                box-shadow: 0 0 0 5px var(--accent-soft);
            }

            .hero h1 {
                margin: .75rem 0 .55rem;
                color: var(--ink);
                font-size: clamp(2.5rem, 5vw, 4.5rem);
                line-height: .98;
                letter-spacing: -.065em;
                font-weight: 780;
            }

            .hero h1 span { color: var(--accent); }

            .hero p {
                max-width: 700px;
                margin: 0;
                color: var(--muted);
                font-size: 1.05rem;
                line-height: 1.7;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(27, 44, 34, .035);
            }

            [data-testid="stMetricLabel"] { color: var(--muted); }
            [data-testid="stMetricValue"] { color: var(--ink); }

            .class-legend {
                display: flex;
                flex-wrap: wrap;
                gap: .55rem;
                margin: .8rem 0 1.4rem;
            }

            .class-chip {
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                padding: .4rem .7rem .4rem .42rem;
                border: 1px solid var(--line);
                border-radius: 999px;
                background: white;
                color: #49564e;
                font-size: .82rem;
            }

            .class-chip b {
                display: grid;
                place-items: center;
                width: 1.35rem;
                height: 1.35rem;
                border-radius: 50%;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: .72rem;
            }

            .section-heading {
                margin: .4rem 0 .2rem;
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 730;
                letter-spacing: -.025em;
            }

            .section-copy {
                margin-bottom: 1rem;
                color: var(--muted);
                font-size: .92rem;
            }

            [data-baseweb="tab-list"] {
                gap: .35rem;
                border-bottom: 1px solid var(--line);
            }

            [data-baseweb="tab"] {
                height: 3rem;
                border-radius: 9px 9px 0 0;
                padding: 0 1rem;
            }

            [data-baseweb="tab-highlight"] { background-color: var(--accent); }

            .stButton > button {
                min-height: 2.8rem;
                border-radius: 10px;
                border: 0;
                background: var(--accent);
                color: white;
                font-weight: 680;
                box-shadow: 0 7px 18px rgba(22, 128, 93, .18);
            }

            .stButton > button:hover {
                background: #106e4f;
                color: white;
                border: 0;
            }

            .prediction-card {
                padding: 1.1rem 1.2rem;
                border: 1px solid #b8decf;
                border-radius: 14px;
                background: var(--accent-soft);
                margin: .35rem 0 1rem;
            }

            .prediction-card small {
                display: block;
                color: #527062;
                font-size: .74rem;
                font-weight: 720;
                letter-spacing: .09em;
                text-transform: uppercase;
            }

            .prediction-card strong {
                display: block;
                margin-top: .25rem;
                color: #0e684b;
                font-size: 1.45rem;
                letter-spacing: -.025em;
            }

            [data-testid="stGraphVizChart"] {
                padding: 1rem;
                border: 1px solid var(--line);
                border-radius: 14px;
                background: white;
                box-shadow: 0 8px 24px rgba(27, 44, 34, .035);
            }

            @media (max-width: 700px) {
                .block-container { padding-top: 1rem; }
                .hero h1 { font-size: 2.65rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_styles()

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="brand-mark">Y</span>Tree Lab</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Build the tree")
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
        st.caption(DATASET_DESCRIPTIONS[dataset_name])
        max_depth = st.slider(
            "Maximum depth",
            min_value=1,
            max_value=10,
            value=5,
            help="Deeper trees model more detail, but become harder to interpret.",
        )
        st.divider()
        st.markdown("**How it works**")
        st.caption(
            "The model tests binary splits and chooses the one with the largest "
            "reduction in Gini impurity at each node."
        )
        st.caption("Built from scratch with Python, NumPy, and pandas.")

    features, labels, label_count, target_names = load_data(
        dataset_name, DATASETS[dataset_name]
    )

    with st.spinner("Finding the best splits..."):
        tree = train_tree(features, labels, max_depth)

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow"><span class="eyebrow-dot"></span>From-scratch CART classifier</div>
            <h1>Decision Tree <span>Visualizer</span></h1>
            <p>See how Gini impurity turns a dataset into a sequence of clear,
            explainable decisions—then trace any sample from root to prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Dataset", dataset_name)
    metric_columns[1].metric("Samples", f"{len(features):,}")
    metric_columns[2].metric("Features", len(features.columns))
    metric_columns[3].metric("Classes", len(label_count))

    if target_names is not None:
        st.markdown(class_legend(target_names), unsafe_allow_html=True)

    overview_tab, prediction_tab = st.tabs(["Tree overview", "Test a sample"])

    with overview_tab:
        st.markdown('<div class="section-heading">Learned decision tree</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-copy">Trained with a maximum depth of {max_depth}. '
            "Follow True to the left and False to the right.</div>",
            unsafe_allow_html=True,
        )
        st.graphviz_chart(to_graphviz(tree), width="stretch")

    with prediction_tab:
        st.markdown('<div class="section-heading">Trace a prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Start from the dataset mean, edit any feature, '
            "and see the exact route the model takes.</div>",
            unsafe_allow_html=True,
        )

        sample = pd.DataFrame([features.mean()], columns=features.columns)
        sample = st.data_editor(
            sample,
            hide_index=True,
            width="stretch",
            key=f"sample-{dataset_name}",
        )

        if st.button("Trace prediction", type="primary", width="stretch"):
            row = sample.iloc[0]
            prediction = classify(row, tree)
            prediction_name = (
                target_names[int(prediction)] if target_names is not None else prediction
            )
            st.markdown(
                '<div class="prediction-card"><small>Predicted class</small>'
                f"<strong>{escape(str(prediction_name))}</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-heading">Highlighted decision path</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">Green nodes and branches show how this sample '
                "moves through the tree.</div>",
                unsafe_allow_html=True,
            )
            st.graphviz_chart(
                highlight_graph(tree, row),
                width="stretch",
            )
        else:
            st.info("Edit the row above, then trace it through the model.", icon="💡")


if __name__ == "__main__":
    main()
