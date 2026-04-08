import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("💳 Credit Scoring System Using Machine Learning")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("User dataset loaded successfully!")
else:
    df = pd.read_csv("UCI_Credit_Card.csv")  # default dataset
    st.info("Default dataset loaded")

# Automatically rename important columns
rename_dict = {}
if "AGE" in df.columns: rename_dict["AGE"] = "age"
if "LIMIT_BAL" in df.columns: rename_dict["LIMIT_BAL"] = "credit_limit"
if "SEX" in df.columns: rename_dict["SEX"] = "sex"
if "EDUCATION" in df.columns: rename_dict["EDUCATION"] = "education"
if "MARRIAGE" in df.columns: rename_dict["MARRIAGE"] = "marriage"
if "default.payment.next.month" in df.columns: rename_dict["default.payment.next.month"] = "credit"

df = df.rename(columns=rename_dict)

# Sidebar options
st.sidebar.title("Dashboard")
show_data = st.sidebar.checkbox("Show Dataset")
show_graphs = st.sidebar.checkbox("Show Graphs")

# Show dataset
if show_data:
    st.subheader("Dataset Preview")
    st.write(df.head())

# Graphs (only if column exists)
if show_graphs:
    if "age" in df.columns:
        st.subheader("Age Distribution")
        st.bar_chart(df["age"])
    if "credit" in df.columns:
        st.subheader("Credit Category Count")
        st.bar_chart(df["credit"].value_counts())

# Prepare data for ML if target exists
if "credit" in df.columns:
    # Replace 2 → 0 for older dataset compatibility
    df["credit"] = df["credit"].replace({2: 0})
    
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

    # Show accuracy
    st.subheader("Model Accuracy")
    st.metric("Accuracy", f"{acc*100:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    st.pyplot(fig)

    st.write("This model classifies customers into good or bad credit risk categories.")
else:
    st.warning("Target column (credit/default) not found. Please upload a valid CSV!")