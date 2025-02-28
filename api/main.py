from fastapi import FastAPI, UploadFile, File, Body
import pickle
import json
import pandas as pd
import numpy as np
import threading
import logging
import os
import joblib
from preprocessing import preprocess_data
from image_processing import extract_image_features
from model_loader import load_model

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to store processed data
STRUCTURED_DATA_PATH = "processed_data/stored_data/processed_structured.pkl"
IMAGE_FEATURES_PATH = "processed_data/stored_data/processed_images.pkl"
SCALER_PATH = "scalers/amount_scaler.pkl"

# Ensure storage directories exist
os.makedirs("processed_data/stored_data", exist_ok=True)

# Load model once at startup
model = load_model()

# Thread lock for handling concurrent requests
file_lock = threading.Lock()

@app.post("/preprocess-structured/")
async def process_structured(data: dict = Body(...)):
    """
    Takes raw structured data, processes it, and stores it in a pickle file.
    """
    try:
        logger.info(f"Received Structured Data: {data}")
        processed_features = preprocess_data(data, for_training=False) # ✅ Exclude Condition & Amount
        logger.info(f"Processed features type: {type(processed_features)}, value: {processed_features}, shape: {processed_features.shape}")

        with file_lock:
            # Load existing structured data if available
            if os.path.exists(STRUCTURED_DATA_PATH):
                with open(STRUCTURED_DATA_PATH, "rb") as f:
                    try:
                        existing_features = pickle.load(f)
                        if not isinstance(existing_features, list):
                            existing_features = list(existing_features)
                    except EOFError:
                        existing_features = []
            else:
                existing_features = []

            # Ensure processed_features is a flat list
            processed_features = np.array(processed_features).flatten().tolist()
            
            existing_features.append(processed_features)

            # Save updated data back to file
            with open(STRUCTURED_DATA_PATH, "wb") as f:
                pickle.dump(existing_features, f)

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
        image_path = "api\processed_data\temp_image.jpg"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        features = extract_image_features(image_path)
        logger.info(f"Processed features shape: {features.shape}")
        
        
        with file_lock:
            # Load existing image features if available
            if os.path.exists(IMAGE_FEATURES_PATH):
                with open(IMAGE_FEATURES_PATH, "rb") as f:
                    try:
                        existing_features = pickle.load(f)
                        if not isinstance(existing_features, list):
                            existing_features = list(existing_features)
                    except EOFError:
                        existing_features = []
            else:
                existing_features = []

            # Append new image features
            existing_features.append(features.tolist())

            # Save updated image features back to file
            with open(IMAGE_FEATURES_PATH, "wb") as f:
                pickle.dump(existing_features, f)

        # Remove temporary image after processing
        os.remove(image_path)
        logger.info("Temporary image deleted.")

        return {"message": "Image processed successfully."}
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}


@app.post("/predict-claim/")
async def predict_claim():
    try:
        # Ensure data exists before loading
        if not os.path.exists(STRUCTURED_DATA_PATH) or not os.path.exists(IMAGE_FEATURES_PATH):
            return {"error": "Structured data or image features not found."}

        with file_lock:
            with open(STRUCTURED_DATA_PATH, "rb") as f:
                structured_features = pickle.load(f)

            with open(IMAGE_FEATURES_PATH, "rb") as f:
                image_features = pickle.load(f)

        if len(structured_features) == 0 or len(image_features) == 0:
            return {"error": "No stored structured data or image features available for prediction."}

        # Load the scaler for Amount
        if not os.path.exists(SCALER_PATH):
            return {"error": f"Scaler file not found: {SCALER_PATH}"}

        amount_scaler = joblib.load(SCALER_PATH)

        # Get latest records for prediction
        structured_features = np.array(structured_features[-1]).reshape(1, -1)
        image_features = np.array(image_features[-1]).reshape(1, -1)

        # Make predictions
        condition_pred, amount_pred = model.predict([image_features, structured_features])

        # Convert condition prediction to binary
        condition_pred = int(condition_pred[0] > 0.5)

        # Process Claim Amount only if condition is positive
        if condition_pred == 1:
            amount_pred = np.array(amount_pred).reshape(-1, 1)  # Ensure 2D for scaler
            amount_pred = amount_scaler.inverse_transform(amount_pred)[0, 0]
            amount_pred = np.expm1(amount_pred)  # Undo log transformation
            amount_pred = max(0, amount_pred)  # Ensure non-negative value
        else:
            amount_pred = 0  # No claim amount if condition is 0

        return {"Condition": condition_pred, "Claim_Amount": float(amount_pred)}

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return {"error": f"An error occurred: {str(e)}"}
