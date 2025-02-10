import tensorflow as tf
import os

# Path to the stored trained model
MODEL_PATH = "models/best_model.keras"

def load_model():
    """
    Loads the trained TensorFlow model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model once when script is executed
model = load_model()
