# Build stage - install build dependencies and train model
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_training/ /app/model_training/
COPY pyproject.toml .
COPY data/ /app/data/

RUN mkdir -p models && \
    python -m model_training.modeling.train

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN pip install --no-cache-dir joblib scikit-learn

COPY --from=builder /app/models/ /app/models/

COPY --from=builder /app/model_training/ /app/model_training/

ENV PYTHONPATH=/app

CMD ["ls", "-la", "models/"] 