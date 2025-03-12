from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import shutil

# Define paths
CSV_SOURCE_PATH = "/opt/airflow/data/raw_data/retrain_data.csv"
CSV_DESTINATION_PATH = "/opt/airflow/data/retrain_data.csv"

IMAGE_SOURCE_DIR = "/opt/airflow/data/images/retrain_images"  # Directory containing new images
IMAGE_DESTINATION_DIR = "/opt/airflow/data/images/test_images"  # Final storage directory

def move_csv_file():
    """Move retrain_data.csv to raw folder, avoiding duplication."""
    try:
        os.makedirs(os.path.dirname(CSV_DESTINATION_PATH), exist_ok=True)  # Ensure directory exists

        if os.path.exists(CSV_SOURCE_PATH):  # Ensure file exists
            if not os.path.exists(CSV_DESTINATION_PATH):  # Move only if not already present
                shutil.move(CSV_SOURCE_PATH, CSV_DESTINATION_PATH)
                print(f"Moved CSV: {CSV_SOURCE_PATH} → {CSV_DESTINATION_PATH}")
            else:
                print(f"Skipped CSV (Already Exists): {CSV_DESTINATION_PATH}")
        else:
            print(f"CSV file not found: {CSV_SOURCE_PATH}")

    except Exception as e:
        print(f"Error moving CSV: {e}")

def move_images():
    """Move new images from source directory to destination, avoiding duplicates."""
    try:
        os.makedirs(IMAGE_DESTINATION_DIR, exist_ok=True)

        if not os.path.exists(IMAGE_SOURCE_DIR) or not os.listdir(IMAGE_SOURCE_DIR):
            print("No new images to move.")
            return

        for file_name in os.listdir(IMAGE_SOURCE_DIR):
            source_image_path = os.path.join(IMAGE_SOURCE_DIR, file_name)
            destination_image_path = os.path.join(IMAGE_DESTINATION_DIR, file_name)

            if os.path.isfile(source_image_path) and not os.path.exists(destination_image_path):
                shutil.move(source_image_path, destination_image_path)  # Move instead of copy
                print(f"Moved Image: {file_name} → {IMAGE_DESTINATION_DIR}")
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

# trigger_processing = TriggerDagRunOperator(
#     task_id='trigger_data_processing',
#     trigger_dag_id='data_processing_dag',
#     wait_for_completion=True,  # Ensures sequential execution
#     dag=dag
# )

# DAG execution order
ingest_csv_task >> ingest_images_task
