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
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_values": df.isna().sum(),
                "missing_%": (df.isna().mean() * 100).round(2),
            }
        ),
        use_container_width=True,
    )

# ----------------------------------------------------------------------
# Page: Exploratory Analysis
# ----------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    if region_col and sales_col:
        st.subheader(f"Sales by {region_col}")
        region_sales = (
            df.groupby(region_col)[sales_col].sum().sort_values(ascending=False).head(15)
        )
        fig = px.bar(
            region_sales,
            x=region_sales.index,
            y=region_sales.values,
            labels={"x": region_col, "y": "Total Sales"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if category_col and sales_col:
        st.subheader(f"Sales by {category_col}")
        cat_sales = (
            df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).head(15)
        )
        fig2 = px.bar(
            cat_sales,
            x=cat_sales.values,
            y=cat_sales.index,
            orientation="h",
            labels={"x": "Total Sales", "y": category_col},
        )
        st.plotly_chart(fig2, use_container_width=True)

    if shipping_col and target_col:
        st.subheader("Late Delivery Risk by Shipping Mode")
        risk_by_ship = df.groupby(shipping_col)[target_col].mean().sort_values(ascending=False)
        fig3 = px.bar(
            risk_by_ship,
            x=risk_by_ship.index,
            y=risk_by_ship.values,
            labels={"x": shipping_col, "y": "Late Delivery Rate"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap (numeric columns)")
        sample_cols = numeric_cols[:15]  # keep it readable
        fig4, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[sample_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig4)
        plt.close(fig4)

# ----------------------------------------------------------------------
# Page: Late Delivery Prediction
# ----------------------------------------------------------------------
elif page == "Late Delivery Prediction":
    st.title("Late Delivery Risk Prediction")

    if target_col is None:
        st.warning(
            "Could not find a 'Late_delivery_risk' column in this dataset, "
            "so the prediction demo is unavailable."
        )
        st.stop()

    st.write(
        "This model uses a Random Forest classifier trained on the dataset "
        "to predict whether an order is at risk of late delivery."
    )

    # --- Feature selection ---
    candidate_features = [
        c
        for c in [
            shipping_col,
            region_col,
            category_col,
            sales_col,
            get_column(df, "Order Item Quantity"),
            get_column(df, "Order Item Discount Rate"),
            get_column(df, "Days for shipping (real)"),
            get_column(df, "Days for shipment (scheduled)"),
        ]
        if c is not None
    ]

    if len(candidate_features) < 2:
        st.warning("Not enough usable feature columns were found for training.")
        st.stop()

    model_df = df[candidate_features + [target_col]].dropna().copy()

    # Encode categorical columns
    encoders = {}
    for col in model_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    X = model_df[candidate_features]
    y = model_df[target_col]

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("Number of trees (n_estimators)", 50, 150, 100, 25)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        with st.spinner("Training Random Forest model..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, n_jobs=2, max_depth=15
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        st.success("Model trained successfully!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
        c2.metric("Precision", f"{precision_score(y_test, preds):.3f}")
        c3.metric("Recall", f"{recall_score(y_test, preds):.3f}")
        c4.metric("F1 Score", f"{f1_score(y_test, preds):.3f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=candidate_features).sort_values(
            ascending=False
        )
        fig2 = px.bar(importance, x=importance.values, y=importance.index, orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Full Classification Report"):
            st.text(classification_report(y_test, preds))
