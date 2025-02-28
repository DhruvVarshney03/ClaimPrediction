# Use a slim Python image to reduce size
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application files efficiently
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
