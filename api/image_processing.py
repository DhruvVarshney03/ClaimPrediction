import numpy as np
import cv2
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Base directory for processed data
PROCESSED_DATA_DIR = "processed_data"

# Paths for storing image features (stored separately from structured data)
IMAGE_FEATURES_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "stored_data", "processed_images.pkl")

# Ensure storage directory exists
os.makedirs(os.path.dirname(IMAGE_FEATURES_FILE_PATH), exist_ok=True)  # Fixed directory creation

# Load pre-trained ResNet50 model (without top layers)
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

IMG_SIZE = (224, 224)  # ResNet50 requires 224x224 images

def preprocess_image(image_path):
    """
    Loads and preprocesses an image:
    - Reads the image from the given path.
    - Resizes it to 224x224 (as required by ResNet50).
    - Normalizes pixel values.
    - Applies ResNet50-specific preprocessing.
    """
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    img_processed = preprocess_input(img_batch)
    
    return img_processed

def extract_image_features(image_path):
    """
    Extracts 2048-dimensional feature vector from the image using ResNet50.
    - Loads and preprocesses the image.
    - Extracts deep features using ResNet50.
    - Flattens the output and stores it.
    """
    img_processed = preprocess_image(image_path)
    features = base_model.predict(img_processed)
    features_flattened = features.flatten()
    
    # Load existing image features if the file exists
    if os.path.exists(IMAGE_FEATURES_FILE_PATH):
        with open(IMAGE_FEATURES_FILE_PATH, "rb") as f:
            existing_features = pickle.load(f)

        # Ensure `existing_features` is a list
        if isinstance(existing_features, np.ndarray):
            existing_features = list(existing_features)  # Fix applied
        elif not isinstance(existing_features, list):
            raise ValueError("Stored data format is not supported!")

        existing_features.append(features_flattened.tolist())
    else:
        existing_features = [features_flattened.tolist()]

    # Store processed image features
    with open(IMAGE_FEATURES_FILE_PATH, "wb") as f:
        pickle.dump(existing_features, f)

    return features_flattened
