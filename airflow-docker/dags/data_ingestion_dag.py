from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import shutil

# Define source and destination paths
TEST_IMAGE_SRC = "images/test_images/"
RAW_IMAGE_DEST = "data/raw/images/"

STRUCTURED_DATA_SRC = "raw_data/"
RAW_STRUCTURED_DATA_DEST = "data/raw/structured_data/"

# Function to move images
def move_images():
    if not os.path.exists(RAW_IMAGE_DEST):
        os.makedirs(RAW_IMAGE_DEST)

    for file in os.listdir(TEST_IMAGE_SRC):
        src = os.path.join(TEST_IMAGE_SRC, file)
        dest = os.path.join(RAW_IMAGE_DEST, file)
        shutil.move(src, dest)

# Function to move structured data (test.csv)
def move_structured_data():
    if not os.path.exists(RAW_STRUCTURED_DATA_DEST):
        os.makedirs(RAW_STRUCTURED_DATA_DEST)

    src_file = os.path.join(STRUCTURED_DATA_SRC, "test.csv")
    dest_file = os.path.join(RAW_STRUCTURED_DATA_DEST, "test.csv")

    if os.path.exists(src_file):
        shutil.move(src_file, dest_file)
    else:
        print(f"File {src_file} not found.")

# DAG definition (manual trigger)
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 11),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_ingestion_dag",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only
    catchup=False,
) as dag:

    ingest_images = PythonOperator(
        task_id="move_images",
        python_callable=move_images,
    )

    ingest_structured_data = PythonOperator(
        task_id="move_structured_data",
        python_callable=move_structured_data,
    )

    ingest_images >> ingest_structured_data  # Ensuring images move first
