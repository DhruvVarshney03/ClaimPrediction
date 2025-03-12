import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import joblib

# Append API path
sys.path.append("/app")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
RAW_CSV_PATH = "/opt/airflow/data/retrain_data.csv"
PROCESSED_DATA_PATH = "/app/api/processed_data/processed_data.pkl"
TRAIN_IMAGES_DIR = "/opt/airflow/data/images/test_images"
ENCODER_PATH = "/app/api/scalers/encoder.pkl"


def load_existing_filenames():
    """Load processed filenames to prevent reprocessing."""
    logging.info("Loading existing processed filenames...")
    return set()


def save_processed_data(data, path, encoder_path, for_training):
    """Save processed data with correct column names."""
    if not data:
        logging.warning(f"No data to save to {path}.")
        return
    
    logging.info("Flattening feature vectors for saving...")
    flattened_data = [[row[0]] + list(row[1]) for row in data]
    
    # Load encoder to get column names
    logging.info(f"Loading encoder from {encoder_path}...")
    encoder = joblib.load(encoder_path)
    categorical_columns = encoder.get_feature_names_out(["Insurance_company"]).tolist()
    
    # Define correct column order
    numerical_columns = ["Cost_of_vehicle", "Min_coverage", "Expiry_date", "Max_coverage"]
    image_feature_columns = [f"feature_{i}" for i in range(2048)]
    
    if for_training:
        logging.info("Including Condition and Amount in processed data.")
        columns = ["Image_path"] + numerical_columns + ["Condition", "Amount"] + categorical_columns + image_feature_columns
    else:
        logging.info("Excluding Condition and Amount for API processing.")
        columns = ["Image_path"] + numerical_columns + categorical_columns + image_feature_columns
    
    processed_df = pd.DataFrame(flattened_data, columns=columns)
    
    logging.info(f"Final processed data shape: {processed_df.shape}")
    joblib.dump(processed_df, path)
    logging.info(f"Saved {len(processed_df)} records to {path}.")


def process_data():
    """Process structured data and images efficiently to reduce memory usage."""
    logging.info("Starting data processing...")
    
    if not os.path.exists(RAW_CSV_PATH):
        logging.info("No new structured data found.")
        return
    
    from api.preprocessing import preprocess_data
    from api.image_processing import extract_image_features
    
    existing_filenames = load_existing_filenames()
    chunk_size = 50  # ✅ Optimized chunk size
    processed_records = []
    
    logging.info(f"Reading structured data from {RAW_CSV_PATH} in chunks of {chunk_size}...")
    for chunk in pd.read_csv(RAW_CSV_PATH, chunksize=chunk_size):
        chunk["Image_filename"] = chunk["Image_path"].apply(os.path.basename)
        chunk = chunk[~chunk["Image_filename"].isin(existing_filenames)]  # Remove already processed
        
        if chunk.empty:
            logging.info("All records in this chunk have already been processed. Skipping...")
            continue
        
        for _, row in chunk.iterrows():
            image_filename = row["Image_filename"]
            full_image_path = os.path.join(TRAIN_IMAGES_DIR, image_filename)
            
            if not os.path.exists(full_image_path):
                logging.warning(f"Skipping {image_filename} (Image not found)")
                continue
            
            try:
                logging.info(f"Processing structured data for {image_filename}...")
                structured_features = preprocess_data(row, for_training=True) 
                logging.info(f"Structured data shape: {structured_features.shape}")
                
                logging.info(f"Extracting image features for {image_filename}...")
                image_features = extract_image_features(full_image_path)
                logging.info(f"Image feature shape: {image_features.shape}")
                
                # ✅ Ensure features are 2D
                structured_features = np.atleast_2d(structured_features)
                image_features = np.atleast_2d(image_features)
                
                combined_features = np.hstack([structured_features, image_features])
                processed_records.append([row["Image_path"]] + combined_features.tolist())
                logging.info(f"Successfully processed {image_filename}.")
            except Exception as e:
                logging.error(f"Error processing {image_filename}: {e}")
    
    # Save processed data
    logging.info("Saving processed data...")
    save_processed_data(processed_records, PROCESSED_DATA_PATH, ENCODER_PATH, for_training=True)
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

def start_task():
    logging.info("Starting the DAG workflow...")

def end_task():
    logging.info("DAG workflow completed successfully.")

start = PythonOperator(
    task_id='start_task',
    python_callable=start_task,
    dag=dag
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)

end = PythonOperator(
    task_id='end_task',
    python_callable=end_task,
    dag=dag
)

# trigger_merge = TriggerDagRunOperator(
#     task_id='trigger_data_merge',
#     trigger_dag_id='data_merge_dag',
#     wait_for_completion=True,
#     dag=dag
# )

start >> process_data_task >> end
