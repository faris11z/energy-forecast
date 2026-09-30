# API image for the day-ahead load forecaster.
# Build from the project root:
#   docker build -t energy-forecast-api .
# Run (data/raw/load_weather.csv must exist locally, it is gitignored):
#   docker run -p 8000:8000 energy-forecast-api

# 3.11 matches a stable wheel set for lightgbm/mlflow.
FROM python:3.11-slim

# Keep the image small and avoid buffering Python output in the container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer first so code edits do not invalidate the pip cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
# Ingested data is not in git, so it has to come from the build context.
COPY data/raw/load_weather.csv ./data/raw/load_weather.csv
COPY models/lgbm_load.txt ./models/lgbm_load.txt

EXPOSE 8000

# Flask's built-in server is fine for a single-container demo.
CMD ["python", "app/api.py"]