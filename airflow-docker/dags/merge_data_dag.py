from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import os
import joblib
import numpy as np
import logging
import pandas as pd
# Paths
FINAL_DATA_PATH = "/app/api/processed_data/final_data.pkl"
PROCESSED_DATA_PATH = "/app/api/processed_data/processed_data.pkl"

# Logging setup
logging.basicConfig(level=logging.INFO)

def merge_data():
    logging.info("Starting data merge process...")

    if not os.path.exists(PROCESSED_DATA_PATH):
        logging.warning("No new processed data found. Skipping merge.")
        return

    new_data = pd.DataFrame(joblib.load(PROCESSED_DATA_PATH))
    logging.info(f"New data shape: {new_data.shape}")

    if os.path.exists(FINAL_DATA_PATH):
        final_data = pd.DataFrame(joblib.load(FINAL_DATA_PATH))
        logging.info(f"Existing final data shape: {final_data.shape}")

        # Ensure column names match
        if list(final_data.columns) != list(new_data.columns):
            logging.error("Column names do not match! Renaming new_data columns to match final_data.")
            new_data.columns = final_data.columns  # Force column names to match
        
        # Ensure column count is the same
        if final_data.shape[1] != new_data.shape[1]:
            logging.error(f"Column mismatch! Final: {final_data.shape[1]}, New: {new_data.shape[1]}")
            common_columns = min(final_data.shape[1], new_data.shape[1])
            final_data = final_data.iloc[:, :common_columns]
            new_data = new_data.iloc[:, :common_columns]
            logging.warning(f"Trimmed datasets to {common_columns} columns.")

        # Backup old data
        backup_path = f"{FINAL_DATA_PATH}.bak"
        joblib.dump(final_data, backup_path)
        logging.info(f"Backup saved at {backup_path}")

        # Merge rows (not columns!)
        merged_data = pd.concat([final_data, new_data], axis=0, ignore_index=True)
    else:
        merged_data = new_data

    joblib.dump(merged_data, FINAL_DATA_PATH)
    logging.info(f"Merged data shape: {merged_data.shape}")

    

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

with DAG('data_merge_dag', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    start_task = DummyOperator(task_id='start')
    
    merge_task = PythonOperator(task_id='merge_data', python_callable=merge_data)
    
    end_task = DummyOperator(task_id='end')

    trigger_retrain = TriggerDagRunOperator(
        task_id='trigger_model_retraining',
        trigger_dag_id='model_retraining_dag',
        wait_for_completion=True  # Ensures sequential execution
    )

    start_task >> merge_task >> end_task >> trigger_retrain