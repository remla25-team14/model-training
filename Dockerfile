# Build stage - install build dependencies and train model
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy lib-ml source code first and install it
COPY lib-ml/ /app/lib-ml/
RUN cd lib-ml && pip install -e .

# Install other dependencies
COPY model-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY model-training/model_training/ /app/model_training/
COPY model-training/pyproject.toml .

# Create data directory structure (training code will handle missing data)
RUN mkdir -p /app/data/raw /app/data/processed /app/data/interim /app/data/external

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