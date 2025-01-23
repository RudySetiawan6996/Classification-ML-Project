import streamlit as st
import requests

st.title("NLP User Prediction")
st.subheader("Enter the data below!")


with st.form(key = "Credit_score_data_form"):
    person_age = st.number_input(
        label = "1.\tEnter person in age:",
        min_value = 17,
        max_value = 60,
        help = "Age accepateble from 17 to 60"
    )
    
    person_income = st.number_input(
        label = "2.\tEnter person income:",
        min_value = 0,
        max_value = 100_000_000_000_000,
        help = "Value range from 0 to 100000000000000"
    )
    
    person_home_ownership = st.selectbox(
        label = "3.\tEnter person_home_ownership status:",
        options = ('RENT', 'OWN', 'MORTGAGE', 'OTHER'),
        help = "The values are RENT', 'OWN', 'MORTGAGE' and 'OTHER'"
    )
    
    person_emp_length = st.number_input(
        label = "4.\tEnter employee length value:",
        min_value = 0.0,
        max_value = 50.0,
        help = "Value range from 0 to 50"
    )
    
    loan_intent = st.selectbox(
        label = "5.\tEnter loan intent:",
        options = ('PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT',
       'DEBTCONSOLIDATION'),
        help = "The values are 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', and 'DEBTCONSOLIDATION'"
    )
    
    loan_grade = st.selectbox(
        label = "6.\tEnter load grade:",
        options = ('A', 'B', 'C', 'D', 'E', 'F', 'G'),
        help = "The values are A until G"
    )
    
    loan_amnt = st.number_input(
        label = "7.\tEnter loan amount value:",
        min_value = 0,
        max_value = 100_000_000_000_000,
        help = "Value range from 0 to 50"
    )
    
    loan_int_rate = st.number_input(
        label = "8.\tEnter loan interest rate value:",
        min_value = 0.0,
        max_value = 100.0,
        help = "The values range from 0 to 100"
    )
    
    loan_percent_income = st.number_input(
        label = "9.\tEnter loan percentage from income value:",
        min_value = 0.0,
        max_value = 1.0,
        help = "The values range from 0 to 1"
    )
    
    cb_person_default_on_file = st.selectbox(
        label = "10.\tEnter person default status on file value:",
        options=('Y','N'),
        help = "The value are Y or N"
    )
    
    cb_person_cred_hist_length = st.number_input(
        label = "11.\tEnter person credit lengt value:",
        min_value = 0.0,
        max_value = 50.0,
        help = "The values range from 0 to 50"
    )
    

    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        raw_data = {
            'person_age':person_age, 
            'person_income':person_income, 
            'person_home_ownership':person_home_ownership,
            'person_emp_length':person_emp_length,
            'loan_intent':loan_intent, 
            'loan_grade':loan_grade, 
            'loan_amnt':loan_amnt,
            'loan_int_rate':loan_int_rate,  
            'loan_percent_income':loan_percent_income,
            'cb_person_default_on_file':cb_person_default_on_file, 
            'cb_person_cred_hist_length':cb_person_cred_hist_length
        }
        
        with st.spinner("Sending data to the API service..."):
            res = requests.post("http://127.0.0.1:8000/predict", json=raw_data).json()
            
        if res["error_msg"]:
            st.error(f"Error: {res['error_msg']}")
        else:
            if res["res"] == "Found API":
                st.success("Prediction Successful!")
                st.write(f"Predicted House Price: {res['house_price_prediction']}")
            else:
                st.error("Prediction Failed!")
