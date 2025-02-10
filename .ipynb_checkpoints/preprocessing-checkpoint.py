import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

# Paths to store processed data
STRUCTURED_DATA_PATH = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\processed_data\stored_data\processed_structured.pkl"
SCALER_PATH = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\feature_scaler.pkl"
ENCODER_PATH = r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\encoder.pkl"

# Ensure storage directory exists
os.makedirs("processed_data", exist_ok=True)

def preprocess_data(raw_data):
    """
    Takes raw structured data (dict format), processes it, and stores it as a pickle file.
    - Converts expiry date to epoch time.
    - Applies one-hot encoding for categorical features.
    - Standardizes numerical features.
    - Scales the `Amount` field using log transformation.
    """

    # Convert dict to DataFrame
    df = pd.DataFrame([raw_data])

    # Ensure required columns exist
    required_cols = ["Insurance_company", "Cost_of_vehicle", "Min_coverage", "Expiry_date", "Max_coverage"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert Expiry Date to epoch time (handle errors)
    df["Expiry_date"] = pd.to_datetime(df["Expiry_date"], format="%d-%m-%Y", errors="coerce")
    if df["Expiry_date"].isna().any():
        raise ValueError("Invalid date format. Use DD-MM-YYYY.")

    df["Expiry_date"] = df["Expiry_date"].astype(int) / 10**9  # Convert to seconds

    # Separate categorical and numerical features
    categorical_features = ["Insurance_company"]
    numerical_features = ["Cost_of_vehicle", "Min_coverage", "Expiry_date", "Max_coverage"]

    # One-Hot Encode categorical variables
    if not os.path.exists(ENCODER_PATH):  # Train encoder if not stored
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded_features = encoder.fit_transform(df[categorical_features])
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)
    else:  # Load existing encoder
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        encoded_features = encoder.transform(df[categorical_features])

    # Standard Scale numerical variables
    if not os.path.exists(SCALER_PATH):  # Train scaler if not stored
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(df[numerical_features])
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    else:  # Load existing scaler
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        scaled_numerical = scaler.transform(df[numerical_features])

    # Combine processed features
    processed_features = np.hstack([encoded_features, scaled_numerical])

    # Store processed structured data
    with open(STRUCTURED_DATA_PATH, "wb") as f:
        pickle.dump(processed_features, f)

    return processed_features
