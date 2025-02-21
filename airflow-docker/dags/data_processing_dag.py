import sys
import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# Append API path
sys.path.append("/app")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Updated paths to match data_ingestion_dag.py
BASE_DIR = os.getenv("AIRFLOW_HOME", "/opt/airflow/")
RAW_CSV_PATH = os.path.join(BASE_DIR, "data/raw_data/retrain_data.csv")
FINAL_DATA_PATH = os.path.join(BASE_DIR, "api/processed_data/final_train_data.pkl")
ROLLBACK_PATH = os.path.join(BASE_DIR, "api/processed_data/rollback.pkl")
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "images/train_images")

def load_existing_data():
    if os.path.exists(FINAL_DATA_PATH):
        with open(FINAL_DATA_PATH, "rb") as f:
            try:
                return pickle.load(f)
            except Exception as e:
                logging.error(f"Failed to load existing data: {e}")
                return []
    return []

def save_rollback_data(last_batch):
    if last_batch:
        with open(ROLLBACK_PATH, "wb") as f:
            pickle.dump(last_batch, f)
        logging.info("Rollback data saved.")

def rollback_last_batch():
    if not os.path.exists(ROLLBACK_PATH):
        logging.warning("No rollback data found.")
        return

    with open(ROLLBACK_PATH, "rb") as f:
        last_batch = pickle.load(f)

    existing_data = load_existing_data()
    new_data = [row for row in existing_data if row not in last_batch]

    with open(FINAL_DATA_PATH, "wb") as f:
        pickle.dump(new_data, f)

    logging.info(f"Rolled back {len(last_batch)} records successfully.")

def process_data():
    if not os.path.exists(RAW_CSV_PATH):
        logging.info("No new structured data found.")
        return

    from api.preprocessing import preprocess_data
    from api.image_processing import extract_image_features

    existing_data = load_existing_data()
    existing_filenames = {os.path.basename(row[0]) for row in existing_data} if existing_data else set()

    new_data = pd.read_csv(RAW_CSV_PATH)
    new_data = new_data.loc[~new_data["Image_path"].apply(os.path.basename).isin(existing_filenames)]

    if new_data.empty:
        logging.info("No new data to process.")
        return

    processed_records = []
    for _, row in new_data.iterrows():
        image_filename = os.path.basename(row["Image_path"])
        full_image_path = os.path.join(TRAIN_IMAGES_DIR, image_filename)

        if not os.path.exists(full_image_path):
            logging.warning(f"Skipping {image_filename} (Image not found)")
            continue

        try:
            structured_features = preprocess_data(row)
            image_features = extract_image_features(full_image_path)
            combined_features = np.hstack([structured_features, image_features])
            processed_records.append([image_filename] + list(combined_features))
            logging.info(f"Processed record for {image_filename}")
        except Exception as e:
            logging.error(f"Error processing {image_filename}: {e}")

    if not processed_records:
        logging.info("No valid records to add.")
        return

    save_rollback_data(processed_records)

    existing_data.extend(processed_records)
    with open(FINAL_DATA_PATH, "wb") as f:
        pickle.dump(existing_data, f)

    logging.info(f"Successfully added {len(processed_records)} new records.")

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_processing_dag',
    default_args=default_args,
    description='DAG for processing structured data and images together',
    schedule_interval='@daily',
    catchup=False
)

# Ensure this DAG runs after data_ingestion_dag
wait_for_ingestion = ExternalTaskSensor(
    task_id="wait_for_data_ingestion",
    external_dag_id="data_ingestion_dag",
    external_task_id="ingest_image_data",
    mode="poke",
    poke_interval=60,  # Check every 60 seconds
    timeout=600,  # Timeout after 10 minutes
    dag=dag
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)

rollback_task = PythonOperator(
    task_id='rollback_data',
    python_callable=rollback_last_batch,
    dag=dag
)

# Execution order
depends_on_past = False
wait_for_ingestion >> process_data_task >> rollback_task
