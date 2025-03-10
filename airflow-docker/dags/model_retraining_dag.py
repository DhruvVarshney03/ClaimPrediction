from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.exceptions import AirflowFailException
from datetime import datetime, timedelta
import os
import logging
import json
import time
import shutil

# Paths
FINAL_DATA_PATH = "/app/api/processed_data/final_data.pkl"
MODEL_DIR = "/app/api/models/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, "backup_best_model.keras")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "models_metadata.json")

# Logging setup
logging.basicConfig(level=logging.INFO)

def fine_tune_model():
    """Fine-tune the existing model with new data."""
    try:
        import joblib
        import numpy as np
        import mlflow
        import mlflow.tensorflow
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        from tensorflow.keras.metrics import AUC

        # (Optional) Enable MLflow autologging for TensorFlow
        mlflow.tensorflow.autolog(log_models=False)

        logging.info("Starting model fine-tuning...")

        if not os.path.exists(FINAL_DATA_PATH):
            logging.error("Final dataset not found. Cannot train.")
            raise AirflowFailException("Dataset missing")

        # Load dataset and convert necessary columns to float32
        data = joblib.load(FINAL_DATA_PATH)
        logging.info(f"Columns in dataset: {list(data.columns)}")

        # Drop unnecessary columns and convert to float32
        structured_data = data.iloc[:, :18].drop(columns=[data.columns[0], data.columns[5], data.columns[6]]).astype(np.float32).values
        image_features = data.iloc[:, 18:].astype(np.float32).values
        condition_labels = data['Condition'].astype(np.float32).values  # Binary classification target
        amount_labels = data['Amount'].values.reshape(-1, 1)  # Regression target

        logging.info(f"Structured Data Shape: {structured_data.shape}")
        logging.info(f"Image Features Shape: {image_features.shape}")
        logging.info(f"Condition Labels Shape: {condition_labels.shape}")
        logging.info(f"Amount Labels Shape: {amount_labels.shape}")

        logging.info(f"Sample structured data: {structured_data[:5]}")
        logging.info(f"Sample image features: {image_features[:5]}")
        logging.info(f"Sample condition labels: {condition_labels[:5]}")
        logging.info(f"Sample amount labels: {amount_labels[:5]}")

        if structured_data.shape[0] != image_features.shape[0]:
            raise AirflowFailException("Mismatch in number of samples between structured and image data.")

        if not os.path.exists(BEST_MODEL_PATH):
            logging.error("Best model not found! Cannot fine-tune.")
            raise AirflowFailException("Best model missing")

        # Load the best existing model
        model = load_model(BEST_MODEL_PATH)
        logging.info(f"Model loaded. Expected Input Shape: {model.input_shape}")

        # Recompile model for fine-tuning with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={"condition_output": 'binary_crossentropy', "amount_output": tf.keras.losses.Huber()},
            metrics={"condition_output": AUC(), "amount_output": "mae"},
        )

        # Setup callbacks for fine-tuning
        model_version = f"fine_tuned_model_{int(time.time())}.keras"
        new_model_path = os.path.join(MODEL_DIR, model_version)
        checkpoint = ModelCheckpoint(new_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

        with mlflow.start_run():
            history = model.fit(
                [image_features, structured_data],
                {"condition_output": condition_labels, "amount_output": amount_labels},
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                callbacks=[checkpoint, early_stopping],
                verbose=1
            )

            # Retrieve key validation metrics
            val_loss = min(history.history["val_loss"])
            # The metric names below assume the model metric naming convention.
            condition_AUC = max(history.history.get("val_condition_output_auc", [0]))
            amount_mae = min(history.history.get("val_amount_output_mae", [float("inf")]))
            logging.info(f"New model validation loss: {val_loss}")
            logging.info(f"New model condition AUC: {condition_AUC}")
            logging.info(f"New model amount MAE: {amount_mae}")

            # Load existing metadata or initialize empty dict
            metadata = {}
            if os.path.exists(MODEL_METADATA_PATH):
                with open(MODEL_METADATA_PATH, "r") as f:
                    metadata = json.load(f)

            previous_best_loss = min(metadata.values()) if metadata else float("inf")
            logging.info(f"Previous best validation loss: {previous_best_loss}")

            # Save new model only if it outperforms the previous best
            if val_loss < previous_best_loss:
                if os.path.exists(BEST_MODEL_PATH):
                    shutil.move(BEST_MODEL_PATH, BACKUP_MODEL_PATH)
                    logging.info(f"Backup of old best model saved at: {BACKUP_MODEL_PATH}")

                metadata[new_model_path] = val_loss
                with open(MODEL_METADATA_PATH, "w") as f:
                    json.dump(metadata, f, indent=4)

                shutil.move(new_model_path, BEST_MODEL_PATH)
                mlflow.tensorflow.log_model(model, artifact_path="best_model")
                mlflow.log_metric("val_loss", val_loss)
                mlflow.log_metric("condition_AUC", condition_AUC)
                mlflow.log_metric("amount_MAE", amount_mae)
                logging.info(f"Fine-tuned model saved as new best at: {BEST_MODEL_PATH}")
            else:
                logging.info("New model did not outperform the previous best. Keeping the old model.")

    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise AirflowFailException("Fine-tuning failed")

# DAG Definition
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

with DAG('model_retraining_dag', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    start_task = DummyOperator(task_id='start')
    fine_tune_task = PythonOperator(task_id='fine_tune_model', python_callable=fine_tune_model)
    end_task = DummyOperator(task_id='end')

    start_task >> fine_tune_task >> end_task
