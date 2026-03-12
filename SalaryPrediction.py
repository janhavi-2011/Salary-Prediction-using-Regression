import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Page Config

st.set_page_config(page_title="Salary Prediction", layout="wide")

st.title(" Salary Prediction Dashboard")

# Sidebar

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset", "Visualization", "Model Comparison", "Predict Salary"]
)

# Load Dataset

data = pd.read_csv("Salary Data.csv")

data = data.dropna()

X = data.drop("Salary", axis=1)
y = data["Salary"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Train Models

lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)


# HOME

if menu == "Home":

    st.header("Project Overview")

    st.write("""
    This project predicts employee salary using **Machine Learning models**.

    Features used in prediction:
    - Age
    - Gender
    - Education Level
    - Job Title
    - Years of Experience
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", data.shape[0])
    col2.metric("Features", data.shape[1]-1)
    col3.metric("Average Salary", f"${int(data['Salary'].mean())}")


# DATASET


elif menu == "Dataset":

    st.header("Dataset Preview")

    st.dataframe(data)

    st.subheader("Statistical Summary")

    st.write(data.describe())


# VISUALIZATION

elif menu == "Visualization":

    st.header("Salary vs Experience")

    fig1 = px.scatter(
        data,
        x="Years of Experience",
        y="Salary",
        color="Education Level",
        title="Salary vs Experience"
    )

    st.plotly_chart(fig1, use_container_width=True)


    st.header("Actual vs Predicted Salary (Linear Regression)")

    actual_vs_pred = pd.DataFrame({
        "Actual Salary": y_test,
        "Predicted Salary": lr_pred
    })

    fig2 = px.scatter(
        actual_vs_pred,
        x="Actual Salary",
        y="Predicted Salary",
        title="Actual vs Predicted Salary"
    )

    st.plotly_chart(fig2, use_container_width=True)


# MODEL COMPARISON


elif menu == "Model Comparison":

    st.header("Model Performance Comparison")

    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    lr_mse = mean_squared_error(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    results = pd.DataFrame({
        "Model":["Linear Regression","Random Forest"],
        "R2 Score":[lr_r2, rf_r2],
        "MSE":[lr_mse, rf_mse]
    })

    st.dataframe(results)

    fig = px.bar(
        results,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig)


# PREDICTION


elif menu == "Predict Salary":

    st.header("Predict Employee Salary")

    age = st.number_input("Age",18,60,30)

    gender = st.selectbox("Gender",data["Gender"].unique())

    education = st.selectbox("Education Level",data["Education Level"].unique())

    job = st.selectbox("Job Title",data["Job Title"].unique())

    experience = st.number_input("Years of Experience",0,40,5)

    model_choice = st.selectbox(
        "Choose Model",
        ["Linear Regression","Random Forest"]
    )

    if st.button("Predict Salary"):

        new_data = pd.DataFrame({
            "Age":[age],
            "Gender":[gender],
            "Education Level":[education],
            "Job Title":[job],
            "Years of Experience":[experience]
        })

        new_data = pd.get_dummies(new_data, drop_first=True)

        new_data = new_data.reindex(columns=X_train.columns, fill_value=0)

        if model_choice == "Linear Regression":
            prediction = lr.predict(new_data)
        else:
            prediction = rf.predict(new_data)


        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
