# Build stage - install build dependencies and train model
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY model_training/ /app/model_training/
COPY pyproject.toml .
COPY data/ /app/data/

# Create directories and train model
RUN mkdir -p models && \
    python -m model_training.modeling.train

# Runtime stage - minimal image with just the trained models
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime dependencies
RUN pip install --no-cache-dir joblib scikit-learn

# Copy trained models from builder stage
COPY --from=builder /app/models/ /app/models/

# Copy minimal runtime code if needed for serving
COPY --from=builder /app/model_training/ /app/model_training/

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["ls", "-la", "models/"] 