import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("💳 Credit Scoring System Using Machine Learning")

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Rename columns (German → English)
df = df.rename(columns={
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

# Sidebar
st.sidebar.title("Dashboard")
show_data = st.sidebar.checkbox("Show Dataset")
show_graphs = st.sidebar.checkbox("Show Graphs")

# Show dataset
if show_data:
    st.subheader("Dataset Preview")
    st.write(df.head())

# Graphs
if show_graphs:
    st.subheader("Age Distribution")
    st.bar_chart(df["age"])

    st.subheader("Credit Category Count")
    st.bar_chart(df["credit"].value_counts())

# Prepare data
df["credit"] = df["credit"].replace({2: 0})

X = df.drop("credit", axis=1)
y = df["credit"]

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

# Simple explanation
st.write("This model classifies customers into good or bad credit risk categories.")