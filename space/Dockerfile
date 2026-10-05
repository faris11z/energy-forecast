# Image for the Hugging Face Space.
#
# Hugging Face clones only this repo into the Space, so the project sources have to
# be pushed alongside it. The deploy workflow does that (see .github/workflows/deploy.yml).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# LightGBM runtime requires libgomp.so.1, which is absent from python:3.11-slim.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify lightgbm can resolve libgomp and import cleanly at build time.
RUN python -c "import lightgbm"

COPY src/ ./src/
COPY app/ ./app/
COPY params.yaml ./
# Both files are gitignored in the project, so the deploy workflow copies them in
# from the build job that produced them.
COPY data/raw/load_weather.csv ./data/raw/load_weather.csv
COPY models/lgbm_load.txt ./models/lgbm_load.txt

EXPOSE 7860

# app/api.py binds to $PORT, which Hugging Face sets to the app_port above.
CMD ["python", "app/api.py"]
