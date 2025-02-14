# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt .
COPY main.py .
COPY preprocessing.py .
COPY image_processing.py .
COPY model_loader.py .
COPY models/ models/
COPY processed_data/ processed_data/
COPY scalers/ scalers/

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which FastAPI will run
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
