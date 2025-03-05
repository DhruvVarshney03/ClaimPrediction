from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_data_drift():
    """Checks data drift between reference data and new batch data."""
    try:

        logging.info("Loading reference and new batch data.")
        reference_data = joblib.load("/app/api/processed_data/final_dataOG.pkl")
        new_data = joblib.load("/app/api/processed_data/processed_data.pkl")

        logging.info(f"Reference Data Shape: {reference_data.shape}")
        logging.info(f"New Data Shape: {new_data.shape}")

        # Log all column names
        logging.info(f"Reference Data Columns: {list(reference_data.columns)}")
        logging.info(f"New Data Columns: {list(new_data.columns)}")

         # Ensure column names match
        if list(reference_data.columns) != list(new_data.columns):
            logging.error("Column names do not match! Renaming new_data columns to match final_data.")
            new_data.columns =reference_data.columns  # Force column names to match
        
        logging.info(f"New Data Columns: {list(new_data.columns)}")
        

        # Drop 'Image_path' if exists
        drop_columns = ["Image_path"]
        reference_data = reference_data.drop(columns=drop_columns, errors="ignore")
        new_data = new_data.drop(columns=drop_columns, errors="ignore")

        drift_results = {}

        for col in reference_data.columns:
            if reference_data[col].dtype in [np.float64, np.int64]:  # Numerical data
                ks_stat, p_value = ks_2samp(reference_data[col], new_data[col])
                drift_results[col] = {"p_value": float(p_value), "drift_detected": bool(p_value < 0.05)}
            else:  # Categorical data
                contingency_table = pd.crosstab(reference_data[col], new_data[col])
                _, p_value, _, _ = chi2_contingency(contingency_table)
                drift_results[col] = {"p_value": float(p_value), "drift_detected": bool(p_value < 0.05)}

        report_path = "/opt/airflow/logs/drift_report.json"
        with open(report_path, "w") as f:
            json.dump(drift_results, f)

        logging.info(f"✅ Data drift report saved at {report_path}")
    except Exception as e:
        logging.error(f"❌ Error in data drift check: {e}")

def check_model_drift():
    
    """Checks model drift by comparing predictions with past performance."""
    try:
        from tensorflow.keras.models import load_model
        from sklearn.metrics import mean_squared_error
        
    
        logging.info("🔄 Loading reference and new batch data.")
        reference_data = joblib.load("/app/api/processed_data/final_dataOG.pkl")
        new_data = joblib.load("/app/api/processed_data/processed_data.pkl")

        logging.info(f"📏 Reference Data Shape: {reference_data.shape}")
        logging.info(f"📏 New Data Shape: {new_data.shape}")

        # Ensure new_data columns match reference_data
        if list(reference_data.columns) != list(new_data.columns):
            logging.warning("⚠️ Column names do not match. Renaming new_data columns to match reference_data.")
            new_data.columns = reference_data.columns

        # Load model
        logging.info("📥 Loading model...")
        model = load_model("/app/api/models/best_model.keras")

        # Ensure proper data selection
        structured_data_new = new_data.iloc[:, :18].drop(columns=[new_data.columns[0], new_data.columns[5], new_data.columns[6]]).astype(np.float32).values
        image_features_new = new_data.iloc[:, 18:].astype(np.float32).values

        structured_data_ref = reference_data.iloc[:, :18].drop(columns=[reference_data.columns[0], reference_data.columns[5], reference_data.columns[6]]).astype(np.float32).values
        image_features_ref = reference_data.iloc[:, 18:].astype(np.float32).values
        
        
        
        # Check shapes before passing to the model
        logging.info(f"Structured Data Shape: {structured_data_new.shape}")
        logging.info(f"Image Features Shape: {image_features_new.shape}")
        

        logging.info("🔮 Generating predictions...")
        new_data["prediction"] = np.array(model.predict([image_features_new, structured_data_new])).flatten()
        reference_data["prediction"] = np.array(model.predict([image_features_ref, structured_data_ref])).flatten()
        

        # Compute MSE and KS test on predictions
        logging.info("📊 Computing MSE and KS statistics...")
        mse_new = mean_squared_error(new_data["prediction"].to_numpy(), new_data["Amount"].to_numpy())
        mse_old = mean_squared_error(reference_data["prediction"].to_numpy(), reference_data["Amount"].to_numpy())
        ks_stat, p_value = ks_2samp(reference_data["prediction"], new_data["prediction"])

        drift_report = {
            "mse_old": mse_old,
            "mse_new": mse_new,
            "mse_drift": abs(mse_old - mse_new),
            "ks_p_value": p_value,
            "drift_detected": p_value < 0.05
        }

        report_path = "/opt/airflow/logs/model_drift_report.json"
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=4)

        logging.info(f"✅ Model drift report saved at {report_path}")
    except Exception as e:
        logging.error(f"❌ Error in model drift check: {e}")

# Define DAG
dag = DAG(
    'governance_drift_monitoring',
    default_args=default_args,
    schedule_interval='@weekly',  # Runs weekly or when manually triggered
    catchup=False,
)

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

data_drift_task = PythonOperator(
    task_id='data_drift_check',
    python_callable=check_data_drift,
    dag=dag,
)

model_drift_task = PythonOperator(
    task_id='model_drift_check',
    python_callable=check_model_drift,
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Task Dependencies
start >> data_drift_task >> model_drift_task >> end
