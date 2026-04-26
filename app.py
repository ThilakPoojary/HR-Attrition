import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Attrition Predictor", layout="wide")

st.title("Employee Attrition Prediction")

st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
monthly_income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 30, 10)

job_role = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative"
])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Single", "Married", "Divorced"
])

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
work_life = st.sidebar.slider("Work Life Balance", 1, 4, 3)

if st.button("Predict"):

    input_dict = {
        'Age': age,
        'DailyRate': daily_rate,
        'MonthlyIncome': monthly_income,
        'DistanceFromHome': distance,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life
    }

    if overtime == "Yes":
        input_dict['OverTime_Yes'] = 1

    if marital_status == "Married":
        input_dict['MaritalStatus_Married'] = 1
    elif marital_status == "Single":
        input_dict['MaritalStatus_Single'] = 1

    if job_role == "Research Scientist":
        input_dict['JobRole_Research Scientist'] = 1
    elif job_role == "Laboratory Technician":
        input_dict['JobRole_Laboratory Technician'] = 1
    elif job_role == "Manufacturing Director":
        input_dict['JobRole_Manufacturing Director'] = 1
    elif job_role == "Healthcare Representative":
        input_dict['JobRole_Healthcare Representative'] = 1

    input_df = pd.DataFrame([input_dict])

    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Employee likely to leave (Probability: {probability:.2f})")
    else:
        st.success(f"Employee likely to stay (Probability: {1-probability:.2f})")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': columns,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(feat_df.set_index("Feature"))