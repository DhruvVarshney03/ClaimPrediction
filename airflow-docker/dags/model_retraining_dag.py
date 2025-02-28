from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import logging
import json
import time

# Paths
FINAL_DATA_PATH = "/app/api/processed_data/final_data.pkl"
MODEL_DIR = "/app/api/models/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "models_metadata.json")
OPTIMIZER_STATE_PATH = os.path.join(MODEL_DIR, "optimizer_weights.pkl")

# Logging setup
logging.basicConfig(level=logging.INFO)

def fine_tune_model():
    """Fine-tune the existing model with new data."""
    import joblib
    import numpy as np
    import mlflow
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    logging.info("Starting model fine-tuning...")

    if not os.path.exists(FINAL_DATA_PATH):
        logging.error("Final dataset not found. Cannot train.")
        return

    # Load the dataset
    data = joblib.load(FINAL_DATA_PATH)
    logging.info(f"All Columns: {list(data.columns)}")

    # Drop unnecessary columns and ensure structured data is numeric
    columns_to_drop = [data.columns[0], data.columns[5], data.columns[6]]
    structured_data = data.iloc[:, :18].drop(columns=columns_to_drop).astype(np.float32).values
    image_features = data.iloc[:, 18:].astype(np.float32).values
    condition_labels = data.iloc[:, -2].astype(np.float32).values
    amount_labels = data.iloc[:, -1].astype(np.float32).values.reshape(-1, 1)

    logging.info(f"Structured Data Shape: {structured_data.shape}, Type: {structured_data.dtype}")
    logging.info(f"Image Features Shape: {image_features.shape}, Type: {image_features.dtype}")
    logging.info(f"Condition Labels Shape: {condition_labels.shape}, Type: {condition_labels.dtype}")
    logging.info(f"Amount Labels Shape: {amount_labels.shape}, Type: {amount_labels.dtype}")

    assert structured_data.shape[0] == image_features.shape[0], "Mismatch in batch size!"

    if not os.path.exists(BEST_MODEL_PATH):
        logging.error("Best model not found! Cannot fine-tune.")
        return

    model = load_model(BEST_MODEL_PATH, compile=False)

    # Restore optimizer state if available
    try:
        if os.path.exists(OPTIMIZER_STATE_PATH):
            with open(OPTIMIZER_STATE_PATH, "rb") as f:
                optimizer_weights = joblib.load(f)
            if model.optimizer is not None:
                model.optimizer.set_weights(optimizer_weights)
                logging.info("Restored optimizer state.")
            else:
                logging.warning("Model optimizer is None. Skipping optimizer state restoration.")
        else:
            logging.warning("Optimizer state not found. Training may not continue seamlessly.")
    except Exception as e:
        logging.error(f"Error restoring optimizer state: {e}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "condition_output": "sparse_categorical_crossentropy",
            "amount_output": "huber_loss",
        },
        metrics={"condition_output": "accuracy", "amount_output": "mae"},
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_version = f"fine_tuned_model_{int(time.time())}.keras"
    new_model_path = os.path.join(MODEL_DIR, model_version)
    checkpoint = ModelCheckpoint(new_model_path, monitor='val_loss', save_best_only=True, mode='min')

    try:
        with mlflow.start_run():
            history = model.fit(
                [image_features, structured_data],
                {'condition_output': condition_labels, 'amount_output': amount_labels},
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )

            val_loss = min(history.history["val_loss"])

            metadata = {}
            if os.path.exists(MODEL_METADATA_PATH):
                with open(MODEL_METADATA_PATH, "r") as f:
                    metadata = json.load(f)

            previous_best_loss = min(metadata.values()) if metadata else float("inf")
            logging.info(f"Previous best validation loss: {previous_best_loss}")
            logging.info(f"New model validation loss: {val_loss}")

            # Save the new model only if it performs better
            if val_loss < previous_best_loss:
                metadata[new_model_path] = val_loss
                with open(MODEL_METADATA_PATH, "w") as f:
                    json.dump(metadata, f, indent=4)

                with open(OPTIMIZER_STATE_PATH, "wb") as f:
                    joblib.dump(model.optimizer.get_weights(), f)
                logging.info("Saved optimizer state.")

                mlflow.tensorflow.log_model(model, artifact_path=new_model_path)
                logging.info(f"Fine-tuned model saved: {new_model_path}")
            else:
                logging.info("New model did not outperform the previous best model. Discarding.")

    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")

def select_best_model():
    """Selects the best model based on validation loss."""
    import shutil

    logging.info("Selecting the best model...")

    if not os.path.exists(MODEL_METADATA_PATH):
        logging.warning("No models available for selection.")
        return

    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if not metadata:
        logging.warning("Model metadata is empty. Keeping the current best model.")
        return

    existing_models = {k: v for k, v in metadata.items() if os.path.exists(k)}

    if not existing_models:
        logging.warning("No valid models found on disk. Keeping the current best model.")
        return

    best_model = min(existing_models, key=existing_models.get)
    best_loss = existing_models[best_model]

    if os.path.abspath(best_model) != os.path.abspath(BEST_MODEL_PATH):
        shutil.copy(best_model, BEST_MODEL_PATH)
        logging.info(f"Best model selected: {best_model} with validation loss: {best_loss}")
    else:
        logging.info(f"Current best model is already the optimal choice: {BEST_MODEL_PATH}")

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

with DAG('model_retraining_dag', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    start_task = DummyOperator(task_id='start')
    fine_tune_task = PythonOperator(task_id='fine_tune_model', python_callable=fine_tune_model)
    select_best_task = PythonOperator(task_id='select_best_model', python_callable=select_best_model)
    end_task = DummyOperator(task_id='end')

    start_task >> fine_tune_task >> select_best_task >> end_task
