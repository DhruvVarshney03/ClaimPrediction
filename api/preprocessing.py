import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory for processed data
PROCESSED_DATA_DIR = "processed_data"

# Paths for structured data and models
STRUCTURED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "stored_data", "processed_structured.pkl")
SCALER_PATH = "/app/api/scalers/feature_scaler.pkl"
ENCODER_PATH = "/app/api/scalers/encoder.pkl"
AMOUNT_SCALER_PATH = "/app/api/scalers/amount_scaler.pkl"

# Ensure directories exist
os.makedirs(os.path.dirname(STRUCTURED_DATA_PATH), exist_ok=True)

# Define feature columns
NUMERICAL_FEATURES = ['Cost_of_vehicle', 'Min_coverage', 'Expiry_date', 'Max_coverage']
CATEGORICAL_FEATURES = ["Insurance_company"]

# Columns required for training (DAG)
TRAINING_COLUMNS = NUMERICAL_FEATURES + ['Condition', 'Amount']  # Includes Condition & Amount

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

    logging.info("Starting data preprocessing...")
    logging.debug(f"Received raw data: {raw_data}")

    # Convert dict to DataFrame
    df = pd.DataFrame([raw_data])

    # **Ensure Expiry Date is converted to epoch time**
    df["Expiry_date"] = pd.to_datetime(df["Expiry_date"], format="%d-%m-%Y", errors="coerce")
    if df["Expiry_date"].isna().any():
        logging.error("Invalid date format detected. Expected format: DD-MM-YYYY.")
        raise ValueError("Invalid date format. Use DD-MM-YYYY.")
    
    df["Expiry_date"] = df["Expiry_date"].astype("int64") // 10**9  # Convert to epoch time

    logging.info("Expiry_date converted to epoch time.")

    # **Select numerical features**
    if for_training:
        numerical_data = df[TRAINING_COLUMNS]  # Includes Amount & Condition
    else:
        numerical_data = df[NUMERICAL_FEATURES]  # Excludes Amount & Condition

    # **Load and apply One-Hot Encoding**
    if not os.path.exists(ENCODER_PATH):
        logging.error(f"Encoder file not found: {ENCODER_PATH}")
        raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")

    with open(ENCODER_PATH, "rb") as f:
        encoder = joblib.load(f)

    try:
        encoded_categorical = encoder.transform(df[CATEGORICAL_FEATURES])
        logging.info(f"One-Hot Encoding successful. Encoded shape: {encoded_categorical.shape}")

    except ValueError as e:
        logging.error("OneHotEncoder encountered an unknown category. Ensure `handle_unknown='ignore'` was set during training.")
        raise e

    # **Load and apply Standard Scaler for numerical features**
    if not os.path.exists(SCALER_PATH):
        logging.error(f"Scaler file not found: {SCALER_PATH}")
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    with open(SCALER_PATH, "rb") as f:
        feature_scaler = joblib.load(f)

    scaled_numerical = feature_scaler.transform(numerical_data[NUMERICAL_FEATURES])
    logging.info("Numerical features scaled successfully.")

    # **Handle Condition separately (Keep as-is for training)**
    if for_training:
        condition_data = df[['Condition']].values  # No scaling, remains 0 or 1

        # **Check for missing values before log transformation**
        if df["Amount"].isna().any():
            logging.warning("Missing values detected in Amount column! Filling with median.")
            df["Amount"].fillna(df["Amount"].median(), inplace=True)  # Handle NaNs before log transformation

        # Apply log transformation to Amount
        df["Amount"] = np.log1p(df["Amount"])  # log(Amount + 1) to avoid log(0) issues
        
        # **Load and apply Amount Scaler**
        if not os.path.exists(AMOUNT_SCALER_PATH):
            logging.error(f"Amount scaler file not found: {AMOUNT_SCALER_PATH}")
            raise FileNotFoundError(f"Amount scaler file not found: {AMOUNT_SCALER_PATH}")
        
        with open(AMOUNT_SCALER_PATH, "rb") as f:
            amount_scaler = joblib.load(f)
        logging.info(f"Loaded Amount Scaler: {amount_scaler}")    
        
        if df["Amount"].isna().any():
            logging.warning("Missing values detected in Amount column!")

        
        # df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        # logging.info(f"Amount column converted to numeric: {df['Amount'].dtype}")


        logging.info(f"Raw Amount values before scaling: {df['Amount'].values}")
        scaled_amount = amount_scaler.transform(df[['Amount']])
        logging.info(f"Scaled Amount values: {scaled_amount}")

        logging.info("Amount feature scaled successfully.")

        # **Combine all features**
        processed_features = np.hstack([scaled_numerical, condition_data, scaled_amount, encoded_categorical])

    else:
        # **Exclude Condition & Amount in API**
        processed_features = np.hstack([scaled_numerical, encoded_categorical])

    logging.info("Processed features generated successfully.")
    logging.info(f"Final processed feature shape: {processed_features.shape}")

    return processed_features
