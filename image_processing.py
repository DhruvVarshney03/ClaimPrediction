import numpy as np
import cv2
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Paths
IMAGE_FEATURES_PATH = "processed_data/stored_data"
FEATURES_FILE_PATH = os.path.join(IMAGE_FEATURES_PATH, "processed_image_features.pkl")  # Fix

# Ensure storage directory exists
os.makedirs(IMAGE_FEATURES_PATH, exist_ok=True)

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
    
    if os.path.exists(FEATURES_FILE_PATH):
        with open(FEATURES_FILE_PATH, "rb") as f:
            existing_features = pickle.load(f)

        if not isinstance(existing_features, list):
            existing_features = existing_features.tolist()

        existing_features.append(features_flattened.tolist())
    else:
        existing_features = [features_flattened.tolist()]

    # Store processed image features
    with open(FEATURES_FILE_PATH, "wb") as f:
        pickle.dump(existing_features, f)  # FIX: Store the full list

    return features_flattened
