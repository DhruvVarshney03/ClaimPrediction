import tensorflow as tf
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Path to the stored trained model
MODEL_PATH = os.path.join("models", "best_model.keras")

def load_model():
    """
    Lazily loads the trained TensorFlow model from disk when called.
    Prevents unnecessary memory usage by not loading at import time.
    """
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    logging.info("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
    return model
