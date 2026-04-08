import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- TITLE ----------------
st.title("💳 Credit Scoring System Using Machine Learning")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Dashboard")
show_data = st.sidebar.checkbox("Show Dataset")
show_graphs = st.sidebar.checkbox("Show Graphs")

# ---------------- FILE UPLOADER ----------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# ---------------- LOAD DATA ----------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("User dataset loaded successfully!")
else:
    df = pd.read_csv("german_credit_data.csv")
    st.info("Default dataset loaded")

# ---------------- RENAME COLUMNS ----------------
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

# ---------------- SHOW DATA ----------------
if show_data:
    st.subheader("Dataset Preview")
    st.write(df.head())

# ---------------- GRAPHS ----------------
if show_graphs:
    st.subheader("Age Distribution")
    st.bar_chart(df["age"])

    st.subheader("Credit Category Count")
    st.bar_chart(df["credit"].value_counts())

# ---------------- DATA CLEANING ----------------
df = df.dropna()

# Convert target column
df["credit"] = df["credit"].replace({2: 0})

# ---------------- FEATURES & TARGET ----------------
X = df.drop("credit", axis=1)
y = df["credit"]

# Convert categorical → numeric
X = pd.get_dummies(X, drop_first=True)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# ---------------- ACCURACY ----------------
st.subheader("Model Accuracy")
st.metric("Accuracy", f"{acc*100:.2f}%")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, pred)

fig = plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
st.pyplot(fig)

# ---------------- INFO ----------------
st.write("This model classifies customers into good or bad credit risk categories.")

# ---------------- WARNING FOR USERS ----------------
st.warning("⚠️ Uploaded dataset must have same structure as original dataset.")