import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Title
# ---------------------------
st.title("💳 Credit Scoring System Using Machine Learning")

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load default dataset
default_df = pd.read_csv("german_credit_data.csv")
# Rename default columns (German → English)
default_df = default_df.rename(columns={
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment",
    "rate": "installment_rate",
    "wohnzeit": "residence_time",
    "alter": "age",
    "kredit": "credit"
})

# ---------------------------
# Handle uploaded dataset
# ---------------------------
if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    st.success("User dataset loaded successfully!")
    
    # Optionally rename common columns (example)
    rename_dict = {}
    if "AGE" in new_df.columns: rename_dict["AGE"] = "age"
    if "LIMIT_BAL" in new_df.columns: rename_dict["LIMIT_BAL"] = "amount"
    if "SEX" in new_df.columns: rename_dict["SEX"] = "sex"
    if "default.payment.next.month" in new_df.columns: rename_dict["default.payment.next.month"] = "credit"
    new_df = new_df.rename(columns=rename_dict)
    
    # Append to default dataset
    df = default_df.append(new_df, ignore_index=True)
else:
    df = default_df
    st.info("Default dataset loaded")

# ---------------------------
# Sidebar options
# ---------------------------
st.sidebar.title("Dashboard")
show_data = st.sidebar.checkbox("Show Dataset")
show_graphs = st.sidebar.checkbox("Show Graphs")

# Show dataset
if show_data:
    st.subheader("Dataset Preview")
    st.write(df.head())

# ---------------------------
# Graphs
# ---------------------------
if show_graphs:
    if "age" in df.columns:
        st.subheader("Age Distribution")
        st.bar_chart(df["age"])
    if "credit" in df.columns:
        st.subheader("Credit Category Count")
        st.bar_chart(df["credit"].value_counts())

# ---------------------------
# Prepare data for ML
# ---------------------------
if "credit" in df.columns:
    df["credit"] = df["credit"].replace({2: 0})  # for old dataset compatibility
    X = df.drop("credit", axis=1)
    y = df["credit"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    # ---------------------------
    # Show accuracy
    # ---------------------------
    st.subheader("Model Accuracy")
    st.metric("Accuracy", f"{acc*100:.2f}%")

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    st.pyplot(fig)

    # ---------------------------
    # Simple explanation
    # ---------------------------
    st.write("This model classifies customers into good or bad credit risk categories.")
else:
    st.warning("Target column (credit/default) not found. Please upload a valid CSV!")