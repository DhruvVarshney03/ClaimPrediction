# Use a specific Airflow version
FROM apache/airflow:latest-python3.11

# Copy the requirements file
COPY airflow.txt .

# Switch to root to install system dependencies
USER root
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y libgl1-mesa-glx git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to Airflow user
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r airflow.txt
