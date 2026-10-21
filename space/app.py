"""Gradio web application for the German Grid (DE-LU) Day-Ahead Load Forecaster.

Runs on Hugging Face Spaces with 100% free Gradio SDK (no Docker required).
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc

# Backward compatibility shim for Hugging Face Hub HfFolder removal
try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                try:
                    from huggingface_hub import get_token
                    return get_token()
                except (ImportError, AttributeError):
                    return None

        huggingface_hub.HfFolder = HfFolder
except (ImportError, AttributeError):
    pass

import gradio as gr
import lightgbm as lgb
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common import MODEL_PATH, RAW_CSV
from features import TARGET, build_features

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CITIES = {
    "berlin": (52.52, 13.41),
    "frankfurt": (50.11, 8.68),
    "munich": (48.14, 11.58),
}
HOURS = 24


def load_model():
    """Load the trained LightGBM booster."""
    # Look in models/ or fallback to current dir
    path = MODEL_PATH if MODEL_PATH.exists() else ROOT / "models" / "lgbm_load.txt"
    if not path.exists():
        path = ROOT / "lgbm_load.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path)), f"Loaded: {path.name}"
    return None, "No model found"


MODEL, MODEL_INFO = load_model()


def fetch_forecast_weather(start: datetime, hours: int) -> pd.DataFrame:
    """Fetch 24h NWP weather forecast averaged over 3 German cities."""
    params = {
        "latitude": ",".join(str(lat) for lat, _ in CITIES.values()),
        "longitude": ",".join(str(lon) for _, lon in CITIES.values()),
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": (start + timedelta(hours=hours)).strftime("%Y-%m-%d"),
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    frames = []
    for city in payload:
        hourly = city["hourly"]
        frames.append(
            pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"], utc=True),
                    "temperature_2m": hourly["temperature_2m"],
                    "wind_speed_10m": hourly["wind_speed_10m"],
                    "shortwave_radiation": hourly["shortwave_radiation"],
                }
            ).set_index("time")
        )
    return pd.concat(frames).groupby(level=0).mean()


def run_forecast():
    """Execute the recursive 24-hour day-ahead forecast and return UI components."""
    if MODEL is None:
        raise gr.Error("Trained LightGBM model not found on disk.")

    now = datetime.now(UTC)
    start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    # Load history
    raw_csv = RAW_CSV if RAW_CSV.exists() else ROOT / "data" / "raw" / "load_weather.csv"
    if not raw_csv.exists():
        raw_csv = ROOT / "load_weather.csv"
    history = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    if history.index.tz is None:
        history.index = history.index.tz_localize("UTC")

    future = fetch_forecast_weather(start, HOURS)
    future = future.reindex(pd.date_range(start=start, periods=HOURS, freq="h", tz="UTC"))
    future[TARGET] = float("nan")

    weather_cols = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]
    frame = pd.concat([history, future[weather_cols + [TARGET]]])

    predictions = []
    for step in range(HOURS):
        step_time = start + timedelta(hours=step)
        feats = build_features(frame)
        row = feats.loc[[step_time]]
        val = float(MODEL.predict(row[feats.columns])[0])
        frame.loc[step_time, TARGET] = val
        predictions.append(
            {
                "Time (UTC)": step_time.strftime("%Y-%m-%d %H:%M"),
                "Hour": step_time.strftime("%H:00"),
                "Forecast Load (MW)": round(val, 1),
                "Temperature (°C)": round(float(row["temperature_2m"].iloc[0]), 1),
                "Wind Speed (m/s)": round(float(row["wind_speed_10m"].iloc[0]), 1),
                "Solar Radiation (W/m²)": round(float(row["shortwave_radiation"].iloc[0]), 1),
            }
        )

    df = pd.DataFrame(predictions)
    loads = df["Forecast Load (MW)"].values
    peak_val = float(loads.max())
    peak_idx = int(loads.argmax())
    min_val = float(loads.min())
    min_idx = int(loads.argmin())
    avg_val = float(loads.mean())

    peak_text = f"**{peak_val:,.1f} MW** (at {df['Hour'].iloc[peak_idx]} UTC)"
    min_text = f"**{min_val:,.1f} MW** (at {df['Hour'].iloc[min_idx]} UTC)"
    avg_text = f"**{avg_val:,.1f} MW** (24h Mean Demand)"

    # Plotly Figure
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Predicted Electricity Demand (MW)", "Weather Drivers (Temperature & Wind)"),
        row_heights=[0.65, 0.35],
    )

    # Main Load trace
    fig.add_trace(
        go.Scatter(
            x=df["Time (UTC)"],
            y=df["Forecast Load (MW)"],
            name="Load Forecast",
            line={"color": "#38bdf8", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            mode="lines+markers",
            marker={"size": 6, "color": "#0284c7"},
        ),
        row=1, col=1,
    )

    # Peak & Min markers
    fig.add_trace(
        go.Scatter(
            x=[df["Time (UTC)"].iloc[peak_idx]],
            y=[peak_val],
            mode="markers+text",
            name="Peak",
            text=[f"Peak: {peak_val:,.0f} MW"],
            textposition="top center",
            marker={"size": 10, "color": "#f59e0b", "symbol": "triangle-up"},
        ),
        row=1, col=1,
    )

    # Temperature trace
    fig.add_trace(
        go.Scatter(
            x=df["Time (UTC)"],
            y=df["Temperature (°C)"],
            name="Temperature (°C)",
            line={"color": "#f43f5e", "width": 2},
        ),
        row=2, col=1,
    )

    # Wind Speed trace
    fig.add_trace(
        go.Scatter(
            x=df["Time (UTC)"],
            y=df["Wind Speed (m/s)"],
            name="Wind Speed (m/s)",
            line={"color": "#10b981", "width": 2, "dash": "dash"},
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=550,
    )
    fig.update_yaxes(title_text="Load (MW)", row=1, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Units", row=2, col=1, gridcolor="#1f2937")
    fig.update_xaxes(gridcolor="#1f2937")

    return peak_text, min_text, avg_text, fig, df


# Gradio UI definition
with gr.Blocks(
    title="German Grid Day-Ahead Energy Forecaster",
    theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        """
        # ⚡ German Grid (DE-LU) Day-Ahead Electricity Load Forecaster
        Real-time 24-hour hourly demand forecasting for Germany & Luxembourg.  
        Powered by **LightGBM**, **DVC-versioned pipelines**, and live **Open-Meteo NWP weather feeds**.
        """
    )

    with gr.Row():
        btn_run = gr.Button("🔄 Refresh Live 24h Forecast", variant="primary", scale=2)

    with gr.Row():
        peak_metric = gr.Markdown("### Peak Load\n--")
        min_metric = gr.Markdown("### Minimum Load\n--")
        avg_metric = gr.Markdown("### 24h Average\n--")
        acc_metric = gr.Markdown("### Model MAE\n**1,499 MW** (-19.5% vs 7d Baseline)")

    with gr.Tabs():
        with gr.TabItem("📊 Forecast Curve & Weather"):
            chart_output = gr.Plot()

        with gr.TabItem("📋 Hourly Breakdown"):
            table_output = gr.DataFrame(
                headers=[
                    "Time (UTC)",
                    "Hour",
                    "Forecast Load (MW)",
                    "Temperature (°C)",
                    "Wind Speed (m/s)",
                    "Solar Radiation (W/m²)",
                ]
            )

        with gr.TabItem("ℹ️ Architecture & MLOps Verification"):
            gr.Markdown(
                """
                ### MLOps Engineering Architecture
                - **Data Ingestion**: 730 days of historical hourly load (Bundesnetzagentur / SMARD) and reanalysis weather.
                - **Target Leakage Control**: Strict 24h and 168h lags; rolling averages are shifted by 1 hour to prevent any lookahead bias.
                - **Split Strategy**: Strict time-based split (last 30 days test, preceding 30 days validation). No random shuffling.
                - **Model**: Regularized LightGBM with 31 leaves and early stopping on validation MAE.
                - **Promotion Gate**: Retrained candidate models must beat production by &ge; 1% MAE to earn the `production` alias.
                - **Drift Monitoring**: Seasonally matched KS-test monitoring comparing the last 30 days to the exact same season one year prior.
                """
            )

    btn_run.click(
        fn=run_forecast,
        inputs=[],
        outputs=[peak_metric, min_metric, avg_metric, chart_output, table_output],
    )

    demo.load(
        fn=run_forecast,
        inputs=[],
        outputs=[peak_metric, min_metric, avg_metric, chart_output, table_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
