import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Base directory for processed data
PROCESSED_DATA_DIR = "processed_data"

# Paths for structured data and models
STRUCTURED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "stored_data", "processed_structured.pkl")
SCALER_PATH = os.path.join("scalers", "feature_scaler.pkl")
ENCODER_PATH = os.path.join("scalers", "encoder.pkl")

# Ensure directories exist
os.makedirs(os.path.dirname(STRUCTURED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

# Define correct feature order (excluding categorical features)
FEATURE_COLUMNS = ['Cost_of_vehicle', 'Min_coverage', 'Expiry_date', 'Max_coverage']
CATEGORICAL_FEATURES = ["Insurance_company"]

def preprocess_data(raw_data):
    """
    Processes structured data for prediction:
    - Converts expiry date to epoch time FIRST.
    - Ensures correct feature order before scaling.
    - One-hot encodes categorical features.
    - Scales numerical features using the same scaler from training.
    """

    print("Received raw data:", raw_data)  # Debugging

    # Convert dict to DataFrame
    df = pd.DataFrame([raw_data])

    # **Ensure Expiry Date is converted to epoch time FIRST**
    df["Expiry_date"] = pd.to_datetime(df["Expiry_date"], format="%d-%m-%Y", errors="coerce")
    if df["Expiry_date"].isna().any():
        raise ValueError("Invalid date format. Use DD-MM-YYYY.")

    df["Expiry_date"] = df["Expiry_date"].astype("int64") // 10**9  # Convert to epoch time safely

    # **Ensure numerical feature order before scaling**
    numerical_data = df[FEATURE_COLUMNS]

    # **Load and apply One-Hot Encoding**
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")
    
    with open(ENCODER_PATH, "rb") as f:
        encoder = joblib.load(f)

    try:
        encoded_categorical = encoder.transform(df[CATEGORICAL_FEATURES])
    except ValueError as e:
        print("OneHotEncoder encountered an unknown category. Ensure `handle_unknown='ignore'` was set during training.")
        raise e

    # **Load and apply Standard Scaler for numerical features**
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    
    with open(SCALER_PATH, "rb") as f:
        feature_scaler = joblib.load(f)

    scaled_numerical = feature_scaler.transform(numerical_data)

    # **Combine categorical & numerical processed features**
    processed_features = np.hstack([scaled_numerical, encoded_categorical])

    # **Store processed structured data efficiently**
    try:
        if os.path.exists(STRUCTURED_DATA_PATH):
            with open(STRUCTURED_DATA_PATH, "rb") as f:
                existing_data = pickle.load(f)

            if not isinstance(existing_data, list):  # Ensure it's a list
                existing_data = existing_data.tolist()

            existing_data.append(processed_features.tolist())  # Append new data
        else:
            existing_data = [processed_features.tolist()]  # Create new list

        with open(STRUCTURED_DATA_PATH, "wb") as f:
            pickle.dump(existing_data, f)

        print("Processed features saved successfully.")  # Debugging
        return processed_features
    
    except Exception as e:
        print(f"Error while saving processed data: {e}")
        return None
