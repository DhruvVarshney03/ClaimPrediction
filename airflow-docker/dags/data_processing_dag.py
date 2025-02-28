import sys
import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import joblib

# Append API path
sys.path.append("/app")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
RAW_CSV_PATH = "/opt/airflow/data/retrain_data.csv"
PROCESSED_DATA_PATH = "/app/api/processed_data/processed_data.pkl"
TRAIN_IMAGES_DIR = "/opt/airflow/data/images/test_images"

def load_existing_filenames():
    """Load processed filenames to prevent reprocessing."""
    return set()

def save_processed_data(data, path):
    """Save processed data as a new file."""
    if not data:
        logging.warning(f"No data to save to {path}.")
        return

    # Flatten feature vectors
    flattened_data = [[row[0]] + list(row[1]) for row in data]  # Convert features from list to individual values
    columns = ["Image_filename"] + [f"feature_{i}" for i in range(len(data[0][1]))]
    processed_df = pd.DataFrame(flattened_data, columns=columns)
    
    joblib.dump(processed_df, path, compress=3)
    logging.info(f"Saved {len(processed_df)} records to {path}.")

def process_data():
    """Process structured data and images efficiently to reduce memory usage."""
    if not os.path.exists(RAW_CSV_PATH):
        logging.info("No new structured data found.")
        return
    
    from api.preprocessing import preprocess_data
    from api.image_processing import extract_image_features
    
    existing_filenames = load_existing_filenames()
    chunk_size = 50  # ✅ Optimized chunk size
    processed_records = []
    
    for chunk in pd.read_csv(RAW_CSV_PATH, chunksize=chunk_size):
        chunk["Image_filename"] = chunk["Image_path"].apply(os.path.basename)
        chunk = chunk[~chunk["Image_filename"].isin(existing_filenames)]  # Remove already processed
        
        if chunk.empty:
            continue
        
        for _, row in chunk.iterrows():
            image_filename = row["Image_filename"]
            full_image_path = os.path.join(TRAIN_IMAGES_DIR, image_filename)

            if not os.path.exists(full_image_path):
                logging.warning(f"Skipping {image_filename} (Image not found)")
                continue
            
            try:
                structured_features = preprocess_data(row, for_training=True) 
                image_features = extract_image_features(full_image_path)
                
                # ✅ Ensure features are 2D
                structured_features = np.atleast_2d(structured_features)
                image_features = np.atleast_2d(image_features)
                
                combined_features = np.hstack([structured_features, image_features])
                processed_records.append([image_filename] + combined_features.tolist())
            except Exception as e:
                logging.error(f"Error processing {image_filename}: {e}")
    
    # Save processed data
    save_processed_data(processed_records, PROCESSED_DATA_PATH)
    logging.info("Data processing completed.")

# DAG Configuration
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
    description='DAG for processing structured data and images efficiently',
    schedule_interval='@daily',
    catchup=False
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)

process_data_task
