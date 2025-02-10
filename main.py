from fastapi import FastAPI, UploadFile, File, Body
import pickle
import json
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from image_processing import extract_image_features
from model_loader import model
import logging
import os
from sklearn.preprocessing import StandardScaler
import joblib

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to store processed data
STRUCTURED_DATA_PATH = "processed_data/stored_data/processed_structured.pkl"
IMAGE_FEATURES_PATH = "processed_data/stored_data/image_features.pkl"
SCALER_PATH = "scalers/amount_scaler.pkl"

# Ensure storage directories exist
os.makedirs("processed_data/stored_data", exist_ok=True)

@app.post("/preprocess-structured/")
async def process_structured(data: dict = Body(...)):
    """
    Takes raw structured data, processes it, and stores it in a pickle file.
    """
    try:
        logger.info(f"Received Structured Data: {data}")  # Log received data
        processed_features = preprocess_data(data)  # Process structured data
        
        if os.path.exists(STRUCTURED_DATA_PATH):
            with open(STRUCTURED_DATA_PATH, "rb") as f:
                existing_data = pickle.load(f)

            # Ensure data format
            if not isinstance(existing_data, list):
                existing_data = existing_data.tolist()

            # Append new record
            existing_data.append(processed_features.tolist())
        else:
            existing_data = [processed_features.tolist()]

        # Save updated structured data
        with open(STRUCTURED_DATA_PATH, "wb") as f:
            pickle.dump(existing_data, f)
        logger.info("Structured data processed and stored successfully.")
        return {"message": "Structured data processed and stored successfully."}
    except Exception as e:
        logger.error(f"Error processing structured data: {str(e)}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """
    Takes an image, extracts features, and stores them.
    """
    try:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        features = extract_image_features(image_path)

        # Append to existing stored image features instead of overwriting
        if os.path.exists(IMAGE_FEATURES_PATH):
            with open(IMAGE_FEATURES_PATH, "rb") as f:
                existing_features = pickle.load(f)

            # Ensure data format
            if not isinstance(existing_features, list):
                existing_features = existing_features.tolist()

        # Append new image feature record
            existing_features.append(features.tolist())
        else:
            existing_features = [features.tolist()]

    # Save updated image features
        with open(IMAGE_FEATURES_PATH, "wb") as f:
            pickle.dump(existing_features, f)

        return {"message": "Image processed successfully."}
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}

@app.post("/predict-claim/")
async def predict_claim():
    """
    Loads stored structured data & image features, makes predictions.
    """
    try:
        # Load structured data
        with open(STRUCTURED_DATA_PATH, "rb") as f:
            structured_features = pickle.load(f)

        # Load image features
        with open(IMAGE_FEATURES_PATH, "rb") as f:
            image_features = pickle.load(f)
        
        # Ensure data is available
        if len(structured_features) == 0 or len(image_features) == 0:
            return {"error": "No stored structured data or image features available for prediction."}

        # Load scaler for Amount
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        amount_scaler = joblib.load(SCALER_PATH)  
            
        if len(structured_features) == 0 or len(image_features) == 0:
            return {"error": "No stored structured data or image features available for prediction."}

        # Ensure both inputs have the correct batch shape
        structured_features = np.array(structured_features[-1]).reshape(1, -1)  # Shape (1, num_features)
        image_features = np.array(image_features[-1]).reshape(1,-1)  # Shape (1, 2048)

        # Make predictions
        condition_pred, amount_pred = model.predict([image_features, structured_features])

        # Convert Condition prediction to 0 or 1 (Assuming sigmoid activation)
        condition_pred = int(condition_pred[0] > 0.5)  # Convert probability to binary

        # Inverse transform the Claim Amount
        amount_pred = amount_scaler.inverse_transform(np.array(amount_pred).reshape(-1, 1))[0, 0]  # Undo scaling
        amount_pred = np.expm1(amount_pred)  # Undo log transformation

        return {"Condition": condition_pred, "Claim_Amount": float(amount_pred)}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}