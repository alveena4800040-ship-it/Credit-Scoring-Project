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
        col_lower = col.lower()
        if col_lower in expected_columns:
            df_renamed.rename(columns={col: expected_columns[col_lower]}, inplace=True)
    return df_renamed

# ---------------- HANDLE USER UPLOAD ----------------
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    user_data = map_columns(user_data)
    st.success("User dataset loaded successfully!")

    # Append safely to main dataset (columns in main dataset)
    combined_data = pd.concat([main_data, user_data], ignore_index=True)
else:
    combined_data = main_data.copy()
    st.info("Default dataset loaded")

# ---------------- AUTO-DETECT CREDIT COLUMN ----------------
possible_target_names = ["credit", "kredit", "Credit", "Kredit"]
credit_col = None
for name in possible_target_names:
    if name in combined_data.columns:
        credit_col = name
        break

if credit_col is None:
    st.error("⚠️ No target column ('credit') found in your dataset. Please include the target column.")
    st.stop()

# Replace 2 with 0 for credit column
combined_data[credit_col] = combined_data[credit_col].replace({2: 0})

# Rename to 'credit' for consistency
if credit_col != "credit":
    combined_data.rename(columns={credit_col: "credit"}, inplace=True)

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
    # Drop target column from user data for prediction
    if "credit" in user_data.columns:
        user_X = user_data.drop("credit", axis=1)
    else:
        user_X = user_data.copy()

    # Convert categorical → numeric
    user_X = pd.get_dummies(user_X, drop_first=True)

    # Add missing columns to match training data
    for col in X_train.columns:
        if col not in user_X.columns:
            user_X[col] = 0
    user_X = user_X[X_train.columns]

    # Make predictions
    user_preds = model.predict(user_X)
    user_data["Prediction"] = user_preds
    st.subheader("Predictions for Uploaded Data")
    st.dataframe(user_data)

# ---------------- INFO ----------------
st.write("This model classifies customers into good or bad credit risk categories.")
st.warning("⚠️ Uploaded dataset can have different column names. The app will map them automatically.")