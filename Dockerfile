FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model training code
COPY model_training/ /app/model_training/
COPY pyproject.toml .

# Copy the data files
COPY data/ /app/data/

# Create necessary directories
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app

# Run the training script
CMD ["python", "-m", "model_training.modeling.train"] 