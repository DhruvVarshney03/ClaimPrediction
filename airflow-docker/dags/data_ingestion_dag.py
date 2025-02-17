from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import shutil

# Define paths
RETRAIN_DATA_FILE = "Fast_Furious_Insured/data/retrain_data.csv"
RAW_DATA_FILE = "Fast_Furious_Insured/raw_data/retrain_data.csv"

RETRAIN_IMAGES_DIR = "Fast_Furious_Insured/images/retrain_images"
TRAIN_IMAGES_DIR = "Fast_Furious_Insured/images/train_images"

def move_csv_file():
    """Move retrain_data.csv to raw folder, avoiding duplication."""
    if os.path.exists(RETRAIN_DATA_FILE):  # Ensure file exists
        if not os.path.exists(RAW_DATA_FILE):  # Move only if not already present
            shutil.move(RETRAIN_DATA_FILE, RAW_DATA_FILE)
            print(f"Moved CSV: {RETRAIN_DATA_FILE} → {RAW_DATA_FILE}")
        else:
            print(f"Skipped CSV (Already Exists): {RAW_DATA_FILE}")

def move_images():
    """Move new images from retrain_images to train_images, avoiding duplicates."""
    if not os.path.exists(TRAIN_IMAGES_DIR):
        os.makedirs(TRAIN_IMAGES_DIR)

    for file_name in os.listdir(RETRAIN_IMAGES_DIR):
        src_path = os.path.join(RETRAIN_IMAGES_DIR, file_name)
        dest_path = os.path.join(TRAIN_IMAGES_DIR, file_name)

        if os.path.isfile(src_path) and not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)
            print(f"Moved Image: {file_name} → {TRAIN_IMAGES_DIR}")
        else:
            print(f"Skipped Image (Already Exists): {file_name}")

# Define default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'data_ingestion_dag',
    default_args=default_args,
    description='DAG to ingest new training data (CSV + images)',
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
ingest_csv_task = PythonOperator(
    task_id='ingest_csv_data',
    python_callable=move_csv_file,
    dag=dag
)

ingest_images_task = PythonOperator(
    task_id='ingest_image_data',
    python_callable=move_images,
    dag=dag
)

# DAG execution order
ingest_csv_task >> ingest_images_task
