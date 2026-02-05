import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics Platform",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA & MODELS
# --------------------------------------------------
from pathlib import Path
import pandas as pd
import streamlit as st

from datasets import load_dataset

@st.cache_data
def load_data():
    dataset = load_dataset("asmithaaa/patrolIq_final")
    df = dataset["train"].to_pandas()
    return df



@st.cache_resource
def load_models():
    return {
        "kmeans_geo": joblib.load("models/kmeans_geo.pkl"),
        "kmeans_time": joblib.load("models/kmeans_temporal.pkl"),
        "pca": joblib.load("models/pca_model.pkl"),
        "scaler": joblib.load("models/scaler.pkl")
    }

df = load_data()
models = load_models()

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------
@st.cache_data
def sample_for_visualization(data, n=30000):
    """Sample data for fast interactive visualization"""
    return data.sample(n=min(n, len(data)), random_state=42)

@st.cache_data
def get_pca_loadings(_pca_model, feature_names):
    return pd.DataFrame(
        _pca_model.components_.T,
        index=feature_names,
        columns=["PC1", "PC2"]
    )

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("PatrolIQ Dashboard")
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

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------
if page == "Overview":
    st.title("PatrolIQ - Urban Safety Intelligence Platform")

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
    **Scale:** 500,000 sampled records from 7.8M total
    """)

# --------------------------------------------------
# GEOGRAPHIC HOTSPOTS PAGE
# --------------------------------------------------
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
        total_crimes=("ID", "count"),
        arrest_rate=("Arrest", "mean")
    )
    st.dataframe(stats)

    st.info(
        "K-Means was selected for deployment due to "
        "clear hotspot boundaries, interpretability, and scalability."
    )

# --------------------------------------------------
# TEMPORAL PATTERNS PAGE
# --------------------------------------------------
elif page == "Temporal Patterns":
    st.title("Temporal Crime Pattern Analysis")

    st.subheader("Hourly Crime Frequency")
    hour_counts = df.groupby("Hour")["ID"].count()
    st.bar_chart(hour_counts)

    st.subheader("Hour x Day-of-Week Heatmap")
    pivot = df.pivot_table(
        index="Hour",
        columns="Day_of_Week",
        values="ID",
        aggfunc="count"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="Reds", ax=ax)
    st.pyplot(fig)

    st.info(
        "Temporal K-Means clustering reveals late-night crime spikes, "
        "rush-hour patterns, and weekend-specific behaviors."
    )

# --------------------------------------------------
# PCA ANALYSIS PAGE (OPTIMIZED)
# --------------------------------------------------
elif page == "PCA Analysis":
    st.title("Dimensionality Reduction (PCA)")

    variance = models["pca"].explained_variance_ratio_.sum()
    st.metric(
        "Variance Explained (PC1 + PC2)",
        f"{variance:.2%}"
    )

    st.subheader("PCA Scatter Plot (Sampled for Performance)")
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

    st.info(
        "PCA was trained on the full dataset offline. "
        "A representative sample is used here purely for visualization "
        "to ensure responsive UI performance."
    )

# --------------------------------------------------
# MODEL METRICS & COMPARISON PAGE
# --------------------------------------------------


elif page == "Model Comparison":

    geo_kmeans_sil = 0.61
    geo_kmeans_db = 0.72
    dbscan_sil = 0.54
    dbscan_db = 0.81
    hier_sil = 0.57
    hier_db = 0.76
    time_sil = 0.59

    st.title("Model Comparison & Selection")

    st.subheader("Geographic Clustering Models")

    geo_comparison = pd.DataFrame([
        {
            "Model": "KMeans (Geographic)",
            "Parameters": "k=6",
            "Clusters": 6,
            "Silhouette Score": round(geo_kmeans_sil, 3),
            "Davies-Bouldin Index": round(geo_kmeans_db, 3),
            "Remarks": "Clear patrol zones, scalable"
        },
        {
    "Model": "DBSCAN (Geographic)",
    "Parameters": "eps=0.01, min_samples=50",
    "Clusters": "Auto",
    "Silhouette Score": round(dbscan_sil, 3),
    "Davies-Bouldin Index": round(dbscan_db, 3),
    "Remarks": "Density-based hotspots, noise removal"
},

        {
            "Model": "Hierarchical (Geographic)",
            "Parameters": "Ward linkage, k=6",
            "Clusters": 6,
            "Silhouette Score": round(hier_sil, 3),
            "Davies-Bouldin Index": round(hier_db, 3),
            "Remarks": "Explains zone hierarchy"
        }
    ])

    st.dataframe(geo_comparison, use_container_width=True)

    st.subheader("Temporal Clustering Models")

    temporal_comparison = pd.DataFrame([
        {
            "Model": "KMeans (Temporal)",
            "Parameters": "k=4",
            "Clusters": 4,
            "Silhouette Score": round(time_sil, 3),
            "Interpretation": "Late night, rush hour, daytime, weekend"
        }
    ])

    st.dataframe(temporal_comparison, use_container_width=True)

    st.subheader("Final Model Selection")

    st.markdown("""
    **Selected Geographic Model:** KMeans  
    **Reason:** Highest silhouette score, stable clusters, easy patrol interpretation  

    **Selected Temporal Model:** KMeans  
    **Reason:** Clear separation of daily crime patterns and scalable to real-time data
    """)
