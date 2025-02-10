from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\models\best_model.keras")
# Load necessary preprocessing objects (scalers, encoders, etc.)
feature_scaler = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\feature_scaler.pkl")
amount_scaler = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\amount_scaler.pkl")
encoder = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\encoder.pkl")
# Configure file upload directory
UPLOAD_FOLDER = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\temp_images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Import the model, preprocess functions, and scalers
# Assuming your model is loaded here
# model = load_model("path_to_your_model")

# File handling: Save the uploaded file securely
def save_file(file):
    #"""Save the uploaded file to the specified folder."""
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

# Image processing: Load and preprocess image
def preprocess_image(image_path):
    #"""Load and preprocess the image for feature extraction."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Example: resizing to 224x224
    image_normalized = image_resized.astype('float32') / 255.0
    return image_normalized

# Extract features: Extract features from preprocessed image
def extract_image_features(image_path):
    """Extract features from the preprocessed image."""
    image_processed = preprocess_image(image_path)
    # Assuming your feature extraction model is preloaded (e.g., ResNet50 or other model)
    features = model.predict(np.expand_dims(image_processed, axis=0))  # Adjust based on your model
    return features.flatten()

# Structured data processing: Process and scale the structured data
def scale_structured_data(structured_data):
    """Scale the structured data using preloaded scalers and one-hot encoding."""
    
    # Convert Expiry_date to epoch time
    structured_data['Expiry_date'] = pd.to_datetime(structured_data['Expiry_date'], format='%d-%m-%Y').astype(int) / 10**9  # Epoch time in seconds
    
    # One-hot encoding for categorical features
    insurance_company_encoded = one_hot_encoder.transform([[structured_data['Insurance_company']]]).toarray()

    # Combine numeric data and encoded categorical data
    numeric_data = [
        structured_data['Cost_of_vehicle'],
        structured_data['Min_coverage'],
        structured_data['Max_coverage'],
        structured_data['Expiry_date']
    ]
    structured_data_combined = np.concatenate([numeric_data, insurance_company_encoded.flatten()])

    # Scale the numeric data
    scaled_data = feature_scaler.transform([structured_data_combined])
    
    return scaled_data


# Prediction: Perform prediction based on image and structured data separately
def make_prediction(image_features, structured_data_scaled):
    """Make a prediction using the loaded model with separate inputs."""
    # Assuming the model accepts both inputs separately (e.g., CNN + Dense)
    condition_pred, amount_pred = model.predict([image_features, structured_data_scaled])
    return condition_pred, amount_pred

# API endpoint: Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Check if both file and input_data are part of the request
    if 'file' not in request.files or 'input_data' not in request.form:
        return jsonify({"error": "File or input_data not found"}), 400

    file = request.files['file']
    input_data = request.form['input_data']

    try:
        # Save the file securely
        image_path = save_file(file)

        # Load and process structured data
        structured_data = json.loads(input_data)

        # Extract image features
        image_features = extract_image_features(image_path)

        # Scale the structured data
        structured_data_scaled = scale_structured_data(structured_data)

        # Ensure both inputs are passed as a list of inputs
        # For the model, pass a list where the first element is image_features and the second is structured_data_scaled
        condition_pred, amount_pred = model.predict([image_features, structured_data_scaled])

        # Return the prediction result
        return jsonify({"condition_prediction": condition_pred.tolist(), "amount_prediction": amount_pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
