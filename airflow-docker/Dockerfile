# Use the official Apache Airflow image
FROM apache/airflow:latest

# Switch to root user before running apt commands
USER root 
RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgl1-mesa-glx

# Switch to the 'airflow' user (already exists in the base image)
USER airflow

# Install mlflow using the 'airflow' user
RUN pip install --no-cache-dir opencv-python-headless tensorflow mlflow


