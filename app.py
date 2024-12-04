import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objs as go
import json
import traceback

app = Flask(__name__)

# Load the model
def load_model():
    try:
        with open('bestmodel.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return None

# Global model variable
MODEL = load_model()

# Feature metadata with more descriptive labels and clearer ranges
FEATURE_METADATA = {
    'yr': {'min': 0, 'max': 1, 'type': 'float', 'label': 'Year (0=First, 1=Second)'},
    'mnth': {'min': 1, 'max': 12, 'type': 'int', 'label': 'Month (1-12)'},
    'holiday': {'min': 0, 'max': 1, 'type': 'int', 'label': 'Holiday (0=No, 1=Yes)'},
    'weekday': {'min': 0, 'max': 6, 'type': 'int', 'label': 'Day of Week (0-6)'},
    'workingday': {'min': 0, 'max': 1, 'type': 'int', 'label': 'Working Day (0=No, 1=Yes)'},
    'weathersit': {'min': 1, 'max': 3, 'type': 'int', 'label': 'Weather (1=Good, 2=Average, 3=Bad)'},
    'hum': {'min': 0, 'max': 1, 'type': 'float', 'label': 'Humidity (0-1)'},
    'windspeed': {'min': 0.0224, 'max': 0.5075, 'type': 'float', 'label': 'Wind Speed (Normalized)'},
    'casual': {'min': 0, 'max': 3500, 'type': 'int', 'label': 'Casual Users (Optional)'},
    'registered': {'min': 0, 'max': 7000, 'type': 'int', 'label': 'Registered Users (Optional)'}
}

@app.route('/')
def index():
    return render_template('index.html', features=FEATURE_METADATA)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()
        
        # Ensure input data matches expected features
        expected_features = [
            'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
            'weathersit', 'hum', 'windspeed', 'casual', 'registered'
        ]
        
        # Convert input to DataFrame with correct order of columns
        input_df = pd.DataFrame([
            [input_data[feature] for feature in expected_features]
        ], columns=expected_features)
        
        # Make prediction
        prediction = MODEL.predict(input_df)[0]
        
        return jsonify({
            'prediction': int(prediction),
        })
    except Exception as e:
        print("Prediction Error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
        }), 400

@app.route('/sensitivity', methods=['POST'])
def sensitivity_analysis():
    try:
        # Get base input and feature to vary
        base_input = request.get_json().get('base_input', {})
        vary_feature = request.get_json().get('vary_feature', 'mnth')
        
        # Validate feature
        if vary_feature not in FEATURE_METADATA:
            return jsonify({'error': 'Invalid feature'}), 400
        
        # Generate sensitivity data
        feature_range = np.linspace(
            FEATURE_METADATA[vary_feature]['min'], 
            FEATURE_METADATA[vary_feature]['max'], 
            20
        )
        
        # Ensure expected features are present
        expected_features = [
            'yr', 'mnth', 'holiday', 'weekday', 'workingday', 
            'weathersit', 'hum', 'windspeed', 'casual', 'registered'
        ]
        
        sensitivity_data = []
        for value in feature_range:
            temp_input = base_input.copy()
            temp_input[vary_feature] = float(value)
            
            # Ensure all features are present
            input_df = pd.DataFrame([
                [temp_input.get(feature, 0) for feature in expected_features]
            ], columns=expected_features)
            
            pred = MODEL.predict(input_df)[0]
            
            sensitivity_data.append({
                'feature_value': value,
                'prediction': int(pred)
            })
        
        return jsonify(sensitivity_data)
    except Exception as e:
        print("Sensitivity Analysis Error:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)