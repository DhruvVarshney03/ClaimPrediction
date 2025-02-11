# Use an official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add this line before installing Python packages
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
# Copy the entire project directory into the container
COPY main.py .
COPY image_processing.py .
COPY preprocessing.py .
COPY model_loader.py .
COPY models/ ./models
COPY scalers/ ./scalers

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
