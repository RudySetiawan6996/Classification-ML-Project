import streamlit as st
import requests

st.title("House Price Prediction")
st.subheader("Enter the data below!")


with st.form(key = "house_price_data_form"):
    area = st.number_input(
        label = "1.\tEnter area in meter value:",
        min_value = 0,
        max_value = 99999,
        help = "Area range from 0 to 99999"
    )
    
    bed_rooms = st.number_input(
        label = "2.\tEnter number of bedrooms value:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )
    
    bath_rooms = st.number_input(
        label = "3.\tEnter number of bathrooms value:",
        min_value = 0,
        max_value = 999,
        help = "Value range from 0 to 999"
    )
    
    stories = st.number_input(
        label = "4.\tEnter number of stories value:",
        min_value = 0,
        max_value = 5,
        help = "Value range from 0 to 5"
    )
    
    main_road = st.selectbox(
        label = "5.\tEnter the status of mainroad value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    guest_room = st.selectbox(
        label = "6.\tEnter the status of guestroom value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    basement = st.selectbox(
        label = "7.\tEnter the status of basement value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    hot_water_heating = st.selectbox(
        label = "8.\tEnter the status of hotwaterheating value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    air_conditioning = st.selectbox(
        label = "9.\tEnter the status of airconditioning value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    parking = st.number_input(
        label = "10.\tEnter number of parking area value:",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10"
    )
    
    pref_area = st.selectbox(
        label = "11.\tEnter the status of prefarea value:",
        options = ("yes", "no"),
        help = "The values are yes or no"
    )
    
    furnishing_status = st.selectbox(
        label = "12.\tEnter the status of furnishingstatus value:",
        options = ("furnished", "semi-furnished", "unfurnished"),
        help = "The values are yes or no"
    )
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        raw_data = {
            "area": area,
            "bedrooms": bed_rooms,
            "bathrooms": bath_rooms,
            "stories": stories,
            "mainroad": main_road,
            "guestroom": guest_room,
            "basement": basement,
            "hotwaterheating": hot_water_heating,
            "airconditioning": air_conditioning,
            "parking": parking,
            "prefarea": pref_area,
            "furnishingstatus": furnishing_status,
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
