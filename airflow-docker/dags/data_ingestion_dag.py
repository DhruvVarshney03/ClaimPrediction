from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import shutil

# Define paths
RETRAIN_DATA_FILE = "/opt/airflow/data/retrain_data.csv"
RAW_DATA_FILE = "/opt/airflow/data/raw_data/retrain_data.csv"

RETRAIN_IMAGES_DIR = "/opt/airflow/images/retrain_images"
TRAIN_IMAGES_DIR = "/opt/airflow/images/train_images"


def move_csv_file():
    """Move retrain_data.csv to raw folder, avoiding duplication."""
    try:
        os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)  # Ensure directory exists

        if os.path.exists(RETRAIN_DATA_FILE):  # Ensure file exists
            if not os.path.exists(RAW_DATA_FILE):  # Move only if not already present
                shutil.move(RETRAIN_DATA_FILE, RAW_DATA_FILE)
                print(f"Moved CSV: {RETRAIN_DATA_FILE} → {RAW_DATA_FILE}")
            else:
                print(f"Skipped CSV (Already Exists): {RAW_DATA_FILE}")
        else:
            print(f"CSV file not found: {RETRAIN_DATA_FILE}")

    except Exception as e:
        print(f"Error moving CSV: {e}")

def move_images():
    """Move new images from retrain_images to train_images, avoiding duplicates."""
    try:
        os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)

        if not os.path.exists(RETRAIN_IMAGES_DIR) or not os.listdir(RETRAIN_IMAGES_DIR):
            print("No new images to move.")
            return

        for file_name in os.listdir(RETRAIN_IMAGES_DIR):
            src_path = os.path.join(RETRAIN_IMAGES_DIR, file_name)
            dest_path = os.path.join(TRAIN_IMAGES_DIR, file_name)

            if os.path.isfile(src_path) and not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)  # Copy before deleting
                os.remove(src_path)
                print(f"Moved Image: {file_name} → {TRAIN_IMAGES_DIR}")
            else:
                print(f"Skipped Image (Already Exists): {file_name}")

    except Exception as e:
        print(f"Error moving images: {e}")

# Define default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),  # Dynamic start date
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
