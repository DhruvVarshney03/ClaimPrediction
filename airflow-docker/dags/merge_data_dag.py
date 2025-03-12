from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from pathlib import Path
import os
import joblib
import pandas as pd
import logging

# Paths
FINAL_DATA_PATH = Path("/app/api/processed_data/final_data.pkl")
PROCESSED_DATA_PATH = Path("/app/api/processed_data/processed_data.pkl")

# Logging setup
logging.basicConfig(level=logging.INFO)

def merge_data():
    try:
        logging.info("🚀 Starting data merge process...")

        if not PROCESSED_DATA_PATH.exists():
            logging.warning("⚠️ No new processed data found. Skipping merge.")
            return

        new_data = pd.DataFrame(joblib.load(PROCESSED_DATA_PATH))
        logging.info(f"📊 New data shape: {new_data.shape}")

        if FINAL_DATA_PATH.exists():
            final_data = pd.DataFrame(joblib.load(FINAL_DATA_PATH))
            logging.info(f"📂 Existing final data shape: {final_data.shape}")

            # Ensure columns match
            if list(final_data.columns) != list(new_data.columns):
                logging.warning("⚠️ Column mismatch detected! Aligning new_data columns with final_data.")
                new_data.columns = final_data.columns  

            # Ensure column count matches
            if final_data.shape[1] != new_data.shape[1]:
                logging.warning(f"⚠️ Column count mismatch! Final: {final_data.shape[1]}, New: {new_data.shape[1]}")
                min_cols = min(final_data.shape[1], new_data.shape[1])
                final_data = final_data.iloc[:, :min_cols]
                new_data = new_data.iloc[:, :min_cols]
                logging.info(f"✅ Trimmed datasets to {min_cols} columns.")

            # Backup old data before merging
            backup_path = FINAL_DATA_PATH.with_suffix(".pkl.bak")
            joblib.dump(final_data, backup_path)
            logging.info(f"🔄 Backup saved at {backup_path}")

            # Merge new rows into final dataset
            merged_data = pd.concat([final_data, new_data], axis=0, ignore_index=True)
        else:
            merged_data = new_data

        joblib.dump(merged_data, FINAL_DATA_PATH)
        logging.info(f"✅ Data merged successfully! Final shape: {merged_data.shape}")

    except Exception as e:
        logging.error(f"❌ Data merge failed due to: {str(e)}", exc_info=True)
        raise  # Ensure DAG fails if merge fails

# Default DAG Arguments
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

# DAG Definition
with DAG('data_merge_dag', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    start_task = DummyOperator(task_id='start')
    
    merge_task = PythonOperator(task_id='merge_data', python_callable=merge_data)
    
    end_task = DummyOperator(task_id='end')

    # trigger_retrain = TriggerDagRunOperator(
    #     task_id='trigger_model_retraining',
    #     trigger_dag_id='model_retraining_dag',
    #     wait_for_completion=True  # Ensures sequential execution
    # )

    start_task >> merge_task >> end_task
