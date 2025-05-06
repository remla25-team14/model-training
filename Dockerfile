# Builder stage
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies without caching
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy necessary files
COPY . .

# Create models directory
RUN mkdir -p models && chmod 777 models

# Run the model training
CMD ["python", "model_train.py"] 