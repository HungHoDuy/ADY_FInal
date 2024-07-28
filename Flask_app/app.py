from flask import Flask, render_template, request, jsonify
import numpy as np
import xgboost as xgb
import pickle
import logging

app = Flask(__name__)

min_max_values = {
    'year': {'min': 1998, 'max': 2019},
    'kilometers_driven': {'min': 171, 'max': 6500000},
    'mileage': {'min': 6, 'max': 67.1},
    'engine': {'min': 72, 'max': 5998},
    'power': {'min': 34.2, 'max': 560},
    'seats': {'min': 5, 'max': 10},
}

locations = ["ahmedabad", "bangalore", "chennai", "coimbatore", "delhi", "hyderabad", "jaipur", "kochi", "kolkata", "mumbai", "pune"]
fuel_types = ["CNG", "diesel", "electric", "lpg", "petrol"]
transmissions = ["manual", "automatic"]
owner_types = ["first", "second", "third", "fourth_and_above"]
brands = ["audi", "ambassador", "bentley", "bmw", "chevrolet", "datsun", "fiat", "force", "ford", "honda", "hyundai", "isuzu", "jaguar", "jeep", "lamborghini", "land_rover", "mahindra", "maruti", "mercedes-benz", "mini", "mitsubishi", "nissan", "porsche", "renault", "skoda", "smart", "tata", "toyota", "volkswagen", "volvo"]

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        # Normalize inputs
        def normalize(value, min_val, max_val):
            if min_val == max_val:
                return 0  # or some other default value, depending on your needs
            return (value - min_val) / (max_val - min_val)

        normalized_inputs = [
            normalize(data['year'], min_max_values['year']['min'], min_max_values['year']['max']),
            normalize(data['kilometers_driven'], min_max_values['kilometers_driven']['min'], min_max_values['kilometers_driven']['max']),
            normalize(data['mileage'], min_max_values['mileage']['min'], min_max_values['mileage']['max']),
            normalize(data['engine'], min_max_values['engine']['min'], min_max_values['engine']['max']),
            normalize(data['power'], min_max_values['power']['min'], min_max_values['power']['max']),
            normalize(data['seats'], min_max_values['seats']['min'], min_max_values['seats']['max']),
        ]

        normalized_inputs += [1 if loc == data['location'] else 0 for loc in locations]
        normalized_inputs += [1 if ft == data['fuel_type'] else 0 for ft in fuel_types]
        normalized_inputs += [1 if tr == data['transmission'] else 0 for tr in transmissions]
        normalized_inputs += [1 if ot == data['owner_type'] else 0 for ot in owner_types]
        normalized_inputs += [1 if br == data['brand'] else 0 for br in brands]

        logging.debug(f"Normalized inputs: {normalized_inputs}")

        # Load the model using pickle
        with open('Models/xgb_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Convert data to numpy array and reshape
        data_array = np.array(normalized_inputs).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data_array)  # Assuming `data_array` is correctly prepared

        factor = 1194.42
        adjusted_prediction = float(prediction[0]) * factor

        # Log the adjusted prediction for debugging
        logging.debug(f"Adjusted Prediction: {adjusted_prediction}")

        # Return the adjusted prediction in JSON format
        return jsonify({'predicted_price': adjusted_prediction})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
