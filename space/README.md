---
title: Energy Forecast API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Energy Forecast API

This folder is the Hugging Face Space definition. `Dockerfile` here is the image
Hugging Face builds; `app_port: 7860` in the frontmatter above is the port the
platform routes traffic to, so the Flask server must bind to `$PORT`.

The Space serves the same `/health` and `/predict` endpoints as the local API:

| endpoint | what it returns |
|----------|-----------------|
| `GET /health`  | status and which model version is being served |
| `GET /predict` | the next 24 hourly load predictions, UTC |

The model is pulled from the MLflow registry using the `production` alias, so
promoting a new model in CI changes what this Space serves without a redeploy.

**Not deployed.** This configuration was written and syntax-checked only; the Space
has to be created and the first push done by hand. See the deploy checklist in the
project README.
