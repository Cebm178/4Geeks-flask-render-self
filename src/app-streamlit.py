import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load the pre-trained model (update the path as per your deployment)
model_path = '/workspaces/4Geeks-flask-render-self/models/KNeighborsRegressor_best_model.sav'
if os.path.exists(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error("Model file not found. Please ensure the correct path.")

# Define the input fields for the user
st.title("Real Estate Price Prediction App")

# Dropdown options for state, city, bedrooms, and bathrooms
state_options = ['California', 'Florida', 'New York', 'Texas']  # Add more options as needed
city_options = {'California': ['Los Angeles', 'San Francisco'], 
                'Florida': ['Miami', 'Orlando'], 
                'New York': ['New York City', 'Buffalo'], 
                'Texas': ['Houston', 'Dallas']}

state = st.selectbox("State", state_options)
city = st.selectbox("City", city_options[state])
bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Number of Bathrooms", [1, 1.5, 2, 2.5, 3, 4])
area = st.number_input("Area in Sqft", min_value=0, step=1)

# Additional variables needed for the model
lot_area = st.number_input("Lot Area in Sqft", min_value=0, step=1)
pp_sq = st.number_input("Price per Square Foot", min_value=0, step=1)
area_per_bedroom = st.number_input("Area per Bedroom", min_value=0, step=1)
market_estimate = st.number_input("Market Estimate", min_value=0.0, step=0.01)
rent_estimate = st.number_input("Rent Estimate", min_value=0.0, step=0.01)
bedroom_to_bathroom_ratio = bedrooms / bathrooms if bathrooms > 0 else 0

# Predict Button
if st.button("Predict Sale Price"):
    if area > 0 and bedrooms > 0 and bathrooms > 0:
        try:
            # Prepare the input data with all required columns (without Zipcode)
            input_data = pd.DataFrame({
                'State': [state],
                'City': [city],
                'Bedroom': [bedrooms],
                'Bathroom': [bathrooms],
                'Area': [area],
                'LotArea': [lot_area],
                'PPSq': [pp_sq],
                'Area_per_Bedroom': [area_per_bedroom],
                'MarketEstimate': [market_estimate],
                'RentEstimate': [rent_estimate],
                'Bedroom_to_Bathroom': [bedroom_to_bathroom_ratio]
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
        st.error("Please ensure that all fields are filled in correctly. Numerical fields must be greater than zero.")