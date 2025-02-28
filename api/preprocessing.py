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
SCALER_PATH = "/app/api/scalers/feature_scaler.pkl"
ENCODER_PATH = "/app/api/scalers/encoder.pkl"
AMOUNT_SCALER_PATH = "/app/api/scalers/amount_scaler.pkl"  # Added for Amount scaling

# Ensure directories exist
os.makedirs(os.path.dirname(STRUCTURED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(AMOUNT_SCALER_PATH), exist_ok=True)

# Define feature columns
NUMERICAL_FEATURES = ['Cost_of_vehicle', 'Min_coverage', 'Expiry_date', 'Max_coverage']
CATEGORICAL_FEATURES = ["Insurance_company"]

# Columns required for training (DAG)
TRAINING_COLUMNS = NUMERICAL_FEATURES + ['Condition', 'Amount']  # Include Condition & Amount

def preprocess_data(raw_data, for_training=False):
    """
    Processes structured data for either training (DAG) or prediction (API).
    
    - Converts expiry date to epoch time FIRST.
    - Ensures correct feature order before scaling.
    - One-hot encodes categorical features.
    - Scales numerical features using the same scaler from training.
    - Keeps Condition as-is.
    - Scales Amount if for_training=True.
    
    Args:
        raw_data (dict): Input structured data.
        for_training (bool): If True, includes Condition & Amount (for DAG). Otherwise, excludes them (for API).
    
    Returns:
        np.array: Processed feature array.
    """

    print("Received raw data:", raw_data)  # Debugging

    # Convert dict to DataFrame
    df = pd.DataFrame([raw_data])

    # **Ensure Expiry Date is converted to epoch time FIRST**
    df["Expiry_date"] = pd.to_datetime(df["Expiry_date"], format="%d-%m-%Y", errors="coerce")
    if df["Expiry_date"].isna().any():
        raise ValueError("Invalid date format. Use DD-MM-YYYY.")
    
    df["Expiry_date"] = df["Expiry_date"].astype("int64") // 10**9  # Convert to epoch time safely

    # **Select numerical features**
    if for_training:
        numerical_data = df[TRAINING_COLUMNS]  # Includes Amount & Condition
    else:
        numerical_data = df[NUMERICAL_FEATURES]  # Excludes Amount & Condition

    # **Load and apply One-Hot Encoding**
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")
    
    with open(ENCODER_PATH, "rb") as f:
        encoder = joblib.load(f)

    try:
        encoded_categorical = encoder.transform(df[CATEGORICAL_FEATURES])
        print(f"Encoded categorical shape: {encoded_categorical.shape}")

    except ValueError as e:
        print("OneHotEncoder encountered an unknown category. Ensure `handle_unknown='ignore'` was set during training.")
        raise e

    # **Load and apply Standard Scaler for numerical features**
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    with open(SCALER_PATH, "rb") as f:
        feature_scaler = joblib.load(f)

    scaled_numerical = feature_scaler.transform(numerical_data[NUMERICAL_FEATURES])

    # **Handle Condition separately (Keep as-is for training)**
    if for_training:
        condition_data = df[['Condition']].values  # No scaling, remains 0 or 1

        # **Load and apply Amount Scaler**
        if not os.path.exists(AMOUNT_SCALER_PATH):
            raise FileNotFoundError(f"Amount scaler file not found: {AMOUNT_SCALER_PATH}")

        with open(AMOUNT_SCALER_PATH, "rb") as f:
            amount_scaler = joblib.load(f)

        scaled_amount = amount_scaler.transform(df[['Amount']])

        # **Combine all features**
        processed_features = np.hstack([scaled_numerical, condition_data, scaled_amount, encoded_categorical])

    else:
        # **Exclude Condition & Amount in API**
        processed_features = np.hstack([scaled_numerical, encoded_categorical])

    print("Processed features generated successfully.")  # Debugging
    print(f"Final processed feature shape: {processed_features.shape}") 
    return processed_features  # Only return, do not save here
