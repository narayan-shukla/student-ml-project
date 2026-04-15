import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("🎓 Student Performance Prediction Dashboard")

st.write("This model predicts student marks based on input features like study hours and attendance.")

# Upload file
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ===== Dataset Preview =====
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ===== Dataset Info =====
    st.subheader("📌 Dataset Information")
    st.write("Shape:", df.shape)
    st.write(df.describe())

    # ===== Target Selection =====
    target = st.selectbox("Select Target Column", df.columns)

    # ===== Graph =====
    st.subheader("📈 Data Visualization")
    fig, ax = plt.subplots()
    df[target].hist(ax=ax)
    ax.set_title(f"Distribution of {target}")
    st.pyplot(fig)

    # ===== Features & Labels =====
    X = df.drop(columns=[target])
    y = df[target]

    # ===== Train-Test Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("⚙️ Model Training")

    # ===== Train Models =====
    lr = LinearRegression()
    rf = RandomForestRegressor()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # ===== Predictions =====
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    # ===== Metrics =====
    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    # ===== Model Comparison =====
    st.subheader("📊 Model Comparison")

    st.write("Linear Regression MSE:", lr_mse)
    st.write("Random Forest MSE:", rf_mse)

    # ===== Best Model =====
    st.subheader("🏆 Best Model")

    if lr_mse < rf_mse:
        st.write("Linear Regression performs better")
    else:
        st.write("Random Forest performs better")

    # ===== Predictions Display =====
    st.subheader("🔮 Sample Predictions")

    results = pd.DataFrame({
        "Actual": y_test.values,
        "LR Prediction": lr_pred,
        "RF Prediction": rf_pred
    })

    st.write(results.head())