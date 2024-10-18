import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pre-trained model (update the path as per your deployment)
model = pickle.load(open('model/real_estate_model.sav', 'rb'))

# Define the input fields for the user
st.title("Real Estate Price Prediction App")

state = st.text_input("State")
city = st.text_input("City")
zipcode = st.text_input("Zipcode", max_chars=5)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
area = st.number_input("Area in Sqft", min_value=0, step=1)

# Predict Button
if st.button("Predict Sale Price"):
    if zipcode and bedrooms > 0 and bathrooms > 0 and area > 0:
        # Prepare the input data
        input_data = pd.DataFrame({
            'state': [state],
            'city': [city],
            'zipcode': [zipcode],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'area': [area]
        })

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        
        # Display the result
        st.success(f"Recommended Sale Price: ${predicted_price:.2f}")
    else:
        st.error("Please fill in all fields correctly.")