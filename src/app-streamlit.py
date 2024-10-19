import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load the pre-trained model (update the path as per your deployment)
model_path = 'model/real_estate_model.sav'
if os.path.exists(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error("Model file not found. Please ensure the correct path.")

# Define the input fields for the user
st.title("Real Estate Price Prediction App")

state = st.text_input("State")
city = st.text_input("City")
zipcode = st.text_input("Zipcode (only digits)", max_chars=5)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
area = st.number_input("Area in Sqft", min_value=0, step=1)

# Predict Button
if st.button("Predict Sale Price"):
    if zipcode.isdigit() and bedrooms > 0 and bathrooms > 0 and area > 0:
        try:
            # Convert zipcode to integer if it's numeric
            zipcode = int(zipcode)

            # Prepare the input data
            input_data = pd.DataFrame({
                'state': [state],
                'city': [city],
                'zipcode': [zipcode],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'area': [area]
            })

            # Debug the input data
            st.write("Input data:", input_data)

            # Make prediction
            predicted_price = model.predict(input_data)[0]
            
            # Display the result
            st.success(f"Recommended Sale Price: ${predicted_price:.2f}")
        except ValueError as ve:
            st.error(f"Value error: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please ensure that all fields are filled in correctly. Zipcode must be digits only, and numerical fields must be greater than zero.")

