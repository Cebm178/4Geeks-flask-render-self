from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
from pickle import load

# Define the Flask app and set the template folder path
app = Flask(__name__, template_folder='../templates')

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/KNeighborsRegressor_best_model.sav")
model = load(open(model_path, "rb"))

# Load the original dataset to find similar properties
df = pd.read_csv('/workspaces/4Geeks-flask-render-self/data/cleaned_df.csv')

# Define home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define predict route
@app.route('/predict', methods=['POST'])
def predict():
    predicted_price = None
    error = None
    
    try:
        # Get form data
        state = request.form['state']
        bedrooms = int(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        area = int(request.form['area'])
        
        # Compute missing features
        bedroom_to_bathroom = bedrooms / bathrooms if bathrooms > 0 else 0
        area_per_bedroom = area / bedrooms if bedrooms > 0 else 0
        
        # Provide default values for other required features
        ppsq = 100  # Default price per square foot, adjust as necessary
        market_estimate = 300000  # Default market estimate, adjust as necessary
        rent_estimate = 1500  # Default rent estimate, adjust as necessary
        lot_area = 5000  # Default lot area, adjust as necessary
        
        # Prepare the input data with all required columns
        input_data = pd.DataFrame({
            'State': [state],
            'Bedroom': [bedrooms],
            'Bathroom': [bathrooms],
            'Area': [area],
            'PPSq': [ppsq],
            'Area_per_Bedroom': [area_per_bedroom],
            'Bedroom_to_Bathroom': [bedroom_to_bathroom],
            'MarketEstimate': [market_estimate],
            'RentEstimate': [rent_estimate],
            'LotArea': [lot_area]
        })
        
        # Debug: check if input data is formatted correctly
        print("Input data for prediction:", input_data)

        # Predict price using the model
        predicted_price = model.predict(input_data)[0]

    except ValueError as ve:
        error = f"Invalid input: {str(ve)}"
        print(error)  # Debugging output

    except Exception as e:
        error = f"An error occurred: {str(e)}"
        print(error)  # Debugging output
    
    return render_template('index.html', predicted_price=predicted_price, error=error)

if __name__ == '__main__':
    app.run(debug=True)
