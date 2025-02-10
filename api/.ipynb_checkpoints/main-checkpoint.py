from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained scalers, encoder, and model
feature_scaler = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\feature_scaler.pkl")
amount_scaler = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\amount_scaler.pkl")
encoder = joblib.load(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\scalers\encoder.pkl")
model = load_model(r"C:\Users\varsh\OneDrive\Desktop\notebook\Fast_Furious_Insured\models\best_model.keras")  # Load the saved model

# Load ResNet model for image feature extraction
resnet_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to preprocess structured data
def preprocess_data(df):
    # Convert Expiry Date to Unix Timestamp
    df['Expiry_date'] = pd.to_datetime(df['Expiry_date'], format='%d-%m-%Y', errors='coerce')
    df['Expiry_date'] = df['Expiry_date'].astype('int64') // 10**9  # Convert to Unix timestamp

    # Define feature columns
    feature_columns = ['Cost_of_vehicle', 'Min_coverage', 'Max_coverage', 'Expiry_date']

    # Apply feature scaling
    df[feature_columns] = feature_scaler.transform(df[feature_columns])

    # Apply log transformation and scale 'Amount'
    df['Amount'] = np.log1p(df['Amount'])  
    df['Amount'] = amount_scaler.transform(df[['Amount']])

    # One-hot encode 'Insurance_company'
    encoded_data = encoder.transform(df[['Insurance_company']]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Insurance_company']))
    
    # Merge encoded data and drop original categorical column
    df_final = df.join(encoded_df).drop(columns=['Insurance_company'])

    return df_final

# Function to extract image features using ResNet50
def extract_image_features(image):
    image = image.resize((224, 224))  # Resize image to match ResNet input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    features = resnet_model.predict(image)  # Extract features
    return features.flatten()  # Convert to 1D array

@app.post("/preprocess-structured/")
async def preprocess_structured(data: dict):
    df = pd.DataFrame([data])  # Convert input dict to DataFrame
    processed_df = preprocess_data(df)  # Apply preprocessing
    return {"processed_data": processed_df.to_dict(orient="records")}

@app.post("/extract-image-features/")
async def extract_image_features_api(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))  # Read uploaded image
    features = extract_image_features(image)  # Extract features
    return {"image_features": features.tolist()}

@app.post("/predict-claim/")
async def predict_claim(structured_data: dict, file: UploadFile = File(...)):
    # Preprocess structured data
    df = pd.DataFrame([structured_data])
    processed_df = preprocess_data(df)

    # Process image
    image = Image.open(io.BytesIO(await file.read()))
    image_features = extract_image_features(image)

    # Combine structured & image features
    combined_features = np.concatenate([processed_df.values.flatten(), image_features])
    combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension

    # Predict using the trained model
    prediction = model.predict(combined_features)

    return {
        "Condition": int(prediction[0][0]),  # Binary condition (0 or 1)
        "Claim_Amount": float(prediction[0][1])  # Predicted claim amount
    }
