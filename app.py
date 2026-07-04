"""
Dataco Smart Supply Chain - ML Analysis Dashboard
--------------------------------------------------
A Streamlit app that explores the DataCo Supply Chain Dataset and
trains a machine-learning model to predict Late Delivery Risk.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dataco Smart Supply Chain - ML Analysis",
    page_icon="📦",
    layout="wide",
)

DATA_FILE = "DataCoSupplyChainDataset.csv"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # DataCo file is often saved with latin-1 encoding
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    return df


def get_column(df: pd.DataFrame, *candidates: str):
    """Return the first matching column name (case-insensitive) or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("📦 Dataco Smart Supply Chain")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory Analysis", "Late Delivery Prediction"],
)

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_FILE}`. Place the dataset CSV in the same "
        "folder as app.py and refresh the page."
    )
    st.stop()

target_col = get_column(df, "Late_delivery_risk", "Late Delivery Risk")
region_col = get_column(df, "Order Region", "Customer Country")
category_col = get_column(df, "Category Name")
sales_col = get_column(df, "Sales", "Order Item Total")
shipping_col = get_column(df, "Shipping Mode")
date_col = get_column(df, "order date (DateOrders)", "Order Date")


# ----------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("Supply Chain Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]}")
    if sales_col:
        col3.metric("Total Sales", f"${df[sales_col].sum():,.0f}")
    if target_col:
        late_pct = df[target_col].mean() * 100
        col4.metric("Late Delivery Rate", f"{late_pct:.1f}%")

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Summary")
    st.dataframe(
