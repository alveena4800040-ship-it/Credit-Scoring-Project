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

# ---------------- LOAD DEFAULT DATA ----------------
main_data = pd.read_csv("german_credit_data.csv")

# ---------------- EXPECTED COLUMN MAPPING ----------------
expected_columns = {
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
}

# ---------------- FUNCTION TO MAP USER COLUMNS ----------------
def map_columns(df):
    df_renamed = df.copy()
    for col in df.columns:
        if col.lower() in expected_columns:  # case-insensitive mapping
            df_renamed.rename(columns={col: expected_columns[col.lower()]}, inplace=True)
    return df_renamed

# ---------------- HANDLE USER UPLOAD ----------------
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    user_data = map_columns(user_data)
    st.success("User dataset loaded successfully!")

    # Drop rows with missing target
    user_data = user_data.dropna(subset=["credit"])

    combined_data = pd.concat([main_data, user_data], ignore_index=True)
else:
    combined_data = main_data.copy()
    st.info("Default dataset loaded")

# ---------------- SHOW DATA ----------------
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(combined_data.head())

# ---------------- GRAPHS ----------------
if show_graphs:
    st.subheader("Age Distribution")
    st.bar_chart(combined_data["age"])

    st.subheader("Credit Category Count")
    st.bar_chart(combined_data["credit"].value_counts())

# ---------------- DATA CLEANING ----------------
combined_data["credit"] = combined_data["credit"].replace({2: 0})

# ---------------- FEATURES & TARGET ----------------
X = combined_data.drop("credit", axis=1)
y = combined_data["credit"]

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

# ---------------- USER PREDICTION ----------------
if uploaded_file is not None:
    user_X = user_data.drop("credit", axis=1)
    user_X = pd.get_dummies(user_X, drop_first=True)

    # Add missing columns to match training data
    for col in X_train.columns:
        if col not in user_X.columns:
            user_X[col] = 0
    user_X = user_X[X_train.columns]

    # Predict
    user_preds = model.predict(user_X)
    user_data["Prediction"] = user_preds
    st.subheader("Predictions for Uploaded Data")
    st.dataframe(user_data)

# ---------------- INFO ----------------
st.write("This model classifies customers into good or bad credit risk categories.")
st.warning("⚠️ Uploaded dataset can have different column names. The app will map them automatically.")