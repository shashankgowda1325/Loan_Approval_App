import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("loan_model.pkl", "rb"))

st.title("Loan Approval Prediction App")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.selectbox("Credit History", [0, 1])

if st.button("Predict"):
    input_data = np.array([[income, loan_amount, credit_history]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
