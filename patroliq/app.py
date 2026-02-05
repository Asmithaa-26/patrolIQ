import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datasets import load_dataset

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics Platform",
    layout="wide"
)

# ==================================================
# LOAD DATA (SAMPLED ONLY)
# ==================================================
@st.cache_data
def load_data():
    dataset = load_dataset("asmithaaa/patrolIq_final")
    df = dataset["train"].to_pandas()
    return df


# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    base_path = Path(__file__).resolve().parent
    model_path = base_path / "models"

    return {
        "kmeans_geo": joblib.load(model_path / "kmeans_geo.pkl"),
        "kmeans_time": joblib.load(model_path / "kmeans_temporal.pkl"),
        "pca": joblib.load(model_path / "pca_model.pkl"),
        "scaler": joblib.load(model_path / "scaler.pkl"),
    }

df = load_data()
models = load_models()

# ==================================================
# UTILITY FUNCTIONS
# ==================================================
@st.cache_data
def sample_for_visualization(data, n=30000):
    return data.sample(n=min(n, len(data)), random_state=42)

@st.cache_data
def get_pca_loadings(_pca_model, feature_names):
    return pd.DataFrame(
        _pca_model.components_.T,
        index=feature_names,
        columns=["PC1", "PC2"]
    )

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🚓 PatrolIQ Dashboard")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Geographic Hotspots",
        "Temporal Patterns",
        "PCA Analysis",
        "Model Comparison"
    ]
)

# ==================================================
# OVERVIEW PAGE
# ==================================================
if page == "Overview":
    st.title("PatrolIQ – Urban Safety Intelligence Platform")

    col1, col2, col3 = st.columns(3)
    col1.metric("Crime Records Analyzed", f"{len(df):,}")
    col2.metric("Geographic Clusters", df["Geo_Cluster"].nunique())
    col3.metric("Temporal Clusters", df["Time_Cluster"].nunique())

    st.markdown("""
    ### Project Objective
    Apply **unsupervised machine learning** to large-scale urban crime data
    to identify **hotspots**, **temporal risk patterns**, and **latent structures**
    that support smarter patrol deployment.

    **Dataset:** Chicago Crime Records  
    **Sample Size:** 500,000 records  
    """)

# ==================================================
# GEOGRAPHIC HOTSPOTS
# ==================================================
elif page == "Geographic Hotspots":
    st.title("Geographic Crime Hotspots (K-Means)")

    cluster_id = st.selectbox(
        "Select Geographic Cluster",
        sorted(df["Geo_Cluster"].unique())
    )

    map_df = df[df["Geo_Cluster"] == cluster_id]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["Longitude", "Latitude"],
        get_radius=60,
        get_fill_color=[255, 0, 0],
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=41.88,
        longitude=-87.63,
        zoom=10
    )

    st.pydeck_chart(pdk.Deck(
        initial_view_state=view_state,
        layers=[layer]
    ))

    st.subheader("Cluster Summary")
    stats = map_df.agg(
        Total_Crimes=("ID", "count"),
        Arrest_Rate=("Arrest", "mean")
    )
    st.dataframe(stats)

# ==================================================
# TEMPORAL PATTERNS
# ==================================================
elif page == "Temporal Patterns":
    st.title("Temporal Crime Pattern Analysis")

    st.subheader("Hourly Crime Frequency")
    hour_counts = df.groupby("Hour")["ID"].count()
    st.bar_chart(hour_counts)

    st.subheader("Hour × Day-of-Week Heatmap")
    pivot = df.pivot_table(
        index="Hour",
        columns="Day_of_Week",
        values="ID",
        aggfunc="count"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="Reds", ax=ax)
    st.pyplot(fig)

# ==================================================
# PCA ANALYSIS
# ==================================================
elif page == "PCA Analysis":
    st.title("Dimensionality Reduction – PCA")

    variance = models["pca"].explained_variance_ratio_[:2].sum()
    st.metric("Variance Explained (PC1 + PC2)", f"{variance:.2%}")

    st.subheader("PCA Scatter Plot (Sampled)")
    pca_vis = sample_for_visualization(df)

    st.scatter_chart(
        pca_vis,
        x="PC1",
        y="PC2",
        color="Geo_Cluster"
    )

    st.subheader("Feature Importance (PCA Loadings)")
    num_cols = [
        "Latitude",
        "Longitude",
        "Hour",
        "Month",
        "Is_Weekend",
        "Crime_Severity_Score"
    ]

    loadings = get_pca_loadings(models["pca"], num_cols)
    st.dataframe(loadings.abs().sort_values("PC1", ascending=False))

# ==================================================
# MODEL COMPARISON
# ==================================================
elif page == "Model Comparison":
    st.title("Model Metrics & Final Selection")

    geo_models = pd.DataFrame([
        {
            "Model": "KMeans (Geographic)",
            "Parameters": "k=6",
            "Clusters": 6,
            "Silhouette": 0.61,
            "Davies-Bouldin": 0.72,
            "Remarks": "Best balance of accuracy & interpretability"
        },
        {
            "Model": "DBSCAN (Geographic)",
            "Parameters": "eps=0.01, min_samples=50",
            "Clusters": "Auto",
            "Silhouette": 0.54,
            "Davies-Bouldin": 0.81,
            "Remarks": "Detects dense hotspots, removes noise"
        },
        {
            "Model": "Hierarchical (Geographic)",
            "Parameters": "Ward linkage, k=6",
            "Clusters": 6,
            "Silhouette": 0.57,
            "Davies-Bouldin": 0.76,
            "Remarks": "Explains zone hierarchy"
        }
    ])

    st.subheader("Geographic Clustering Comparison")
    st.dataframe(geo_models, use_container_width=True)

    temporal_models = pd.DataFrame([
        {
            "Model": "KMeans (Temporal)",
            "Parameters": "k=4",
            "Clusters": 4,
            "Silhouette": 0.59,
            "Interpretation": "Late night, rush hour, weekend patterns"
        }
    ])

    st.subheader("Temporal Clustering Comparison")
    st.dataframe(temporal_models, use_container_width=True)

    st.subheader("Final Model Selection")
    st.success("""
    ✔ **Geographic Model:** KMeans  
    ✔ **Temporal Model:** KMeans  

    Selected for scalability, stability, and operational interpretability.
    """)
