from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import numpy as np
import logging
from api.preprocessing import preprocess_data
from api.image_processing import extract_image_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
RAW_CSV_PATH = "Fast_Furious_Insured/data/retrain_data/retrain_data.csv"
FINAL_DATA_PATH = "Fast_Furious_Insured/api/processed_data/final_train_data.pkl"
ROLLBACK_PATH = "Fast_Furious_Insured/api/processed_data/rollback.pkl"
TRAIN_IMAGES_DIR = "Fast_Furious_Insured/images/retrain_images"

# Load existing dataset (if exists)
def load_existing_data():
    if os.path.exists(FINAL_DATA_PATH):
        with open(FINAL_DATA_PATH, "rb") as f:
            return pickle.load(f)
    return []

# Save rollback data
def save_rollback_data(last_batch):
    with open(ROLLBACK_PATH, "wb") as f:
        pickle.dump(last_batch, f)
    logging.info("Rollback data saved.")

# Rollback last batch
def rollback_last_batch():
    if not os.path.exists(ROLLBACK_PATH):
        logging.warning("No rollback data found.")
        return
    
    # Load rollback records
    with open(ROLLBACK_PATH, "rb") as f:
        last_batch = pickle.load(f)

    # Load existing dataset
    existing_data = load_existing_data()

    # Remove last batch from dataset
    new_data = [row for row in existing_data if row not in last_batch]

    # Save updated dataset
    with open(FINAL_DATA_PATH, "wb") as f:
        pickle.dump(new_data, f)

    logging.info(f"Rolled back {len(last_batch)} records successfully.")

# Process structured data & images together
def process_data():
    """Processes structured data and image features, then stores them together in final_train_data.pkl."""
    if not os.path.exists(RAW_CSV_PATH):
        logging.info("No new structured data found.")
        return

    # Load existing data
    existing_data = load_existing_data()

    # Extract filenames from existing dataset to prevent duplicates
    existing_filenames = {os.path.basename(row[0]) for row in existing_data}  # Extract only filename (not full path)
    
    # Read new structured data
    new_data = pd.read_csv(RAW_CSV_PATH)

    # Remove already processed records (by checking only the filename)
    new_data["Image_Filename"] = new_data["Image_path"].apply(os.path.basename)
    new_data = new_data.loc[~new_data["Image_Filename"].isin(existing_filenames)]
    
    if new_data.empty:
        logging.info("No new data to process.")
        return

    # Process data
    processed_records = []
    
    for _, row in new_data.iterrows():
        image_path = row["Image_path"]
        image_filename = os.path.basename(image_path)
        full_image_path = os.path.join(TRAIN_IMAGES_DIR, image_filename)

        # Check if image exists
        if not os.path.exists(full_image_path):
            logging.warning(f"Skipping {image_filename} (Image not found)")
            continue

        try:
            # Process structured data
            structured_features = preprocess_data(row)

            # Extract image features
            image_features = extract_image_features(full_image_path)

            # Combine structured and image features
            combined_features = np.hstack([structured_features, image_features])

            # Append to dataset
            processed_records.append([image_filename] + list(combined_features))

            logging.info(f"Processed record for {image_filename}")

        except Exception as e:
            logging.error(f"Error processing {image_filename}: {e}")

    if not processed_records:
        logging.info("No valid records to add.")
        return

    # Save rollback data before modifying final dataset
    save_rollback_data(processed_records)

    # Append to final dataset
    existing_data.extend(processed_records)
    with open(FINAL_DATA_PATH, "wb") as f:
        pickle.dump(existing_data, f)

    logging.info(f"Successfully added {len(processed_records)} new records.")

# DAG default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'data_processing_dag',
    default_args=default_args,
    description='DAG for processing structured data and images together',
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
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

# DAG execution order
process_data_task >> rollback_task
