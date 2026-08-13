"""
Real Estate Valuation — EDA & Linear Regression
------------------------------------------------
A beginner-friendly Streamlit app for exploring the UCI Real Estate
Valuation dataset and training a scikit-learn Linear Regression model.

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this repo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Valuation — EDA & Linear Regression",
    page_icon="🏠",
    layout="wide",
)

DEFAULT_DATA_PATH = "data/real_estate_valuation.csv"
TARGET_COL = "Y house price of unit area"
ID_COL = "No"

COLUMN_DESCRIPTIONS = {
    "No": "Row index / ID — not a real feature.",
    "X1 transaction date": "Date of sale in decimal-year format (e.g. 2013.25 ≈ March 2013).",
    "X2 house age": "Age of the house, in years.",
    "X3 distance to the nearest MRT station": "Distance in meters to the nearest metro station.",
    "X4 number of convenience stores": "Number of convenience stores within walking distance.",
    "X5 latitude": "Geographic latitude.",
    "X6 longitude": "Geographic longitude.",
    "Y house price of unit area": "Target — price per unit area (New Taiwan Dollar / Ping).",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


def load_data():
    st.sidebar.header("1. Data")
    source = st.sidebar.radio(
        "Choose a data source",
        ["Use bundled sample dataset", "Upload my own CSV"],
        help="The bundled dataset is the UCI Real Estate Valuation dataset.",
    )
    if source == "Upload my own CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            return pd.read_csv(uploaded), "Uploaded file"
        st.sidebar.info("No file uploaded yet — showing the bundled sample dataset instead.")
        return load_default_data(), "Bundled sample dataset"
    return load_default_data(), "Bundled sample dataset"


# --------------------------------------------------------------------------
# Sidebar — data + model controls
# --------------------------------------------------------------------------
st.title("🏠 Real Estate Valuation")
st.caption(
    "A beginner-friendly walkthrough of EDA and Linear Regression on the "
    "UCI Real Estate Valuation dataset — deployed with Streamlit."
)

df, source_label = load_data()
st.sidebar.success(f"Loaded: {source_label}  ({df.shape[0]} rows, {df.shape[1]} columns)")

st.sidebar.header("2. Model settings")
test_size = st.sidebar.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5) / 100
random_state = st.sidebar.number_input("Random state (for reproducibility)", value=42, step=1)

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
droppable_default = [c for c in [ID_COL] if c in df.columns]
drop_cols = st.sidebar.multiselect(
    "Columns to drop before modeling",
    options=[c for c in df.columns if c != TARGET_COL],
    default=droppable_default,
    help="ID-like columns (e.g. 'No') usually carry no predictive signal and are safe to drop.",
)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📋 Overview", "🔍 EDA", "📈 Model & Evaluation", "🔮 Try a Prediction"]
)

# --------------------------------------------------------------------------
# TAB 1 — Overview
# --------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column descriptions")
    desc_rows = [
        {"Column": c, "Description": COLUMN_DESCRIPTIONS.get(c, "—")} for c in df.columns
    ]
    st.table(pd.DataFrame(desc_rows))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    st.info(
        "Use the sidebar to switch between the bundled sample dataset and your own CSV, "
        "and to control which columns are dropped before modeling."
    )

# --------------------------------------------------------------------------
# TAB 2 — EDA
# --------------------------------------------------------------------------
with tab_eda:
    st.subheader("Summary statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Missing values & duplicates")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum().rename("missing_count"))
    with c2:
        st.metric("Duplicate rows", int(df.duplicated().sum()))

    if TARGET_COL in df.columns:
        st.subheader(f"Distribution of target: `{TARGET_COL}`")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[TARGET_COL].dropna(), bins=30, color="steelblue", edgecolor="black")
        ax.set_xlabel(TARGET_COL)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Distribution of a chosen feature")
    feature_choice = st.selectbox(
        "Pick a numeric column to visualize",
        [c for c in numeric_cols if c != TARGET_COL],
    )
    if feature_choice:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[feature_choice].dropna(), bins=30, color="seagreen", edgecolor="black")
        ax.set_xlabel(feature_choice)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Correlation heatmap")
    corr_cols = [c for c in numeric_cols]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        if TARGET_COL in corr.columns:
            st.write(f"Correlation with `{TARGET_COL}`, sorted:")
            st.dataframe(corr[TARGET_COL].sort_values(ascending=False))

    st.subheader("Scatter plot: feature vs target")
    x_axis = st.selectbox(
        "X-axis feature", [c for c in numeric_cols if c != TARGET_COL], key="scatter_x"
    )
    if x_axis and TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df[x_axis], df[TARGET_COL], alpha=0.5, color="darkorange")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(TARGET_COL)
        st.pyplot(fig)

# --------------------------------------------------------------------------
# TAB 3 — Model & Evaluation
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Train a Linear Regression model")

    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found in this dataset.")
    else:
        model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        feature_cols = [c for c in model_df.columns if c != TARGET_COL and c in numeric_cols]

        st.write(f"**Features used ({len(feature_cols)}):** {', '.join(feature_cols)}")
        if drop_cols:
            st.write(f"**Dropped columns:** {', '.join(drop_cols)}")

        X = model_df[feature_cols]
        y = model_df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("R² score", f"{r2:.3f}")
        m2.metric("RMSE", f"{rmse:.3f}")
        m3.metric("MAE", f"{mae:.3f}")

        st.subheader("Model coefficients")
        coef_df = pd.DataFrame(
            {"Feature": feature_cols, "Coefficient": model.coef_}
        ).sort_values(by="Coefficient", key=abs, ascending=False)
        st.write(f"Intercept: `{model.intercept_:.4f}`")
        st.dataframe(coef_df, use_container_width=True)

        st.subheader("Predicted vs. Actual")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color="teal")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Residual plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_pred, residuals, alpha=0.6, color="crimson")
        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_xlabel("Predicted price")
        ax.set_ylabel("Residual (Actual - Predicted)")
        st.pyplot(fig)

        # Persist trained model + feature columns for the "Try a Prediction" tab
        st.session_state["model"] = model
        st.session_state["feature_cols"] = feature_cols
        st.session_state["X"] = X

# --------------------------------------------------------------------------
# TAB 4 — Try a Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict a price with your own inputs")

    if "model" not in st.session_state:
        st.warning("Train a model first in the 'Model & Evaluation' tab.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["feature_cols"]
        X_ref = st.session_state["X"]

        st.write("Adjust the sliders to match a hypothetical property:")
        input_values = {}
        cols = st.columns(2)
        for i, feat in enumerate(feature_cols):
            col = cols[i % 2]
            min_v = float(X_ref[feat].min())
            max_v = float(X_ref[feat].max())
            mean_v = float(X_ref[feat].mean())
            input_values[feat] = col.slider(
                feat, min_value=min_v, max_value=max_v, value=mean_v
            )

        input_df = pd.DataFrame([input_values])[feature_cols]
        prediction = model.predict(input_df)[0]

        st.metric("Predicted house price per unit area", f"{prediction:.2f}")
        st.caption(
            "This is a simple linear model — treat this as a teaching demo, "
            "not a real valuation tool."
        )

st.divider()
st.caption(
    "Built for beginner practice with the UCI Real Estate Valuation dataset "
    "(Xu, 2018) · scikit-learn Linear Regression · Streamlit"
)
