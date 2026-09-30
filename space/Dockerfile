# Image for the Hugging Face Space.
#
# Hugging Face clones only this repo into the Space, so the project sources have to
# be pushed alongside it. The deploy workflow does that (see .github/workflows/deploy.yml).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
