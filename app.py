import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import os

# --- Paths ---
LINEAR_MODEL_PATH = "model/TCS_Stock_Predictor.pkl"
LSTM_MODEL_PATH = "model/tcs_lstm_model.keras"
SCALER_PATH = "model/tcs_lstm_scaler.pkl"
DATA_PATH = "data/TCS_stock_history.csv"

print("Model Exists?", os.path.exists(LSTM_MODEL_PATH))
print("Scaler Exists?", os.path.exists(SCALER_PATH))
# --- Load Models ---
try:
    lin_model = joblib.load(LINEAR_MODEL_PATH)
    print(f"✅ Linear model loaded from {LINEAR_MODEL_PATH}")
except Exception:
    lin_model = None
    print("❌ Failed to load linear model")

try:
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ LSTM model & scaler loaded")
except Exception:
    lstm_model = None
    scaler = None
    print("❌ Failed to load LSTM model/scaler")


# --- Data Loader ---
def load_df():
    df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df.dropna(subset=["Date"]).sort_values("Date")


# --- Helpers ---
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# --- Plot Functions ---
def plot_trend_volume():
    df = load_df()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(df["Date"], df["Close"], label="Close")
    ax1.plot(df["Date"], df["MA50"], label="MA50")
    ax1.plot(df["Date"], df["MA200"], label="MA200")
    ax1.set_title("Price & MAs"), ax1.legend()
    ax2.plot(df["Date"], df["Volume"], label="Volume")
    ax2.set_title("Volume")
    return fig_to_pil(fig)


def plot_div_splits():
    df = load_df()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df.get("Dividends", 0), label="Dividends")
    ax.plot(df["Date"], df.get("Stock Splits", 0), label="Splits")
    ax.set_title("Dividends & Splits"), ax.legend()
    return fig_to_pil(fig)


def plot_ma_crossover():
    df = load_df()
    df["ShortMA"] = df["Close"].rolling(20).mean()
    df["LongMA"] = df["Close"].rolling(50).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], label="Close")
    ax.plot(df["Date"], df["ShortMA"], label="20‑day MA")
    ax.plot(df["Date"], df["LongMA"], label="50‑day MA")
    ax.set_title("MA Crossover"), ax.legend()
    return fig_to_pil(fig)


def plot_daily_change():
    df = load_df()
    df["DailyChange"] = df["Close"].pct_change() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["DailyChange"].dropna(), bins=50)
    ax.set_title("Daily % Change")
    return fig_to_pil(fig)


def plot_actual_predicted():
    df = load_df()
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)
    feats = [
        "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week", "Month"]
    X = df[feats]
    y_true = df["Close"]
    y_pred = lin_model.predict(X) if lin_model else [0] * len(X)
    mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], y_true, label="Actual")
    ax.plot(df["Date"], y_pred, label="Pred", alpha=0.7)
    ax.set_title(f"Actual vs Predicted 🧮 (MSE={mse:.2f}, R²={r2:.2f})")
    ax.legend()
    return fig_to_pil(fig)


def forecast_lstm():
    if lstm_model is None or scaler is None:
        return Image.new("RGB", (400, 200), "gray")
    df = load_df()
    data = scaler.transform(df[["Close"]])
    seq = data[-60:].reshape(1, 60, 1)
    pred = lstm_model.predict(seq)[0, 0]
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(["Next"], [pred])
    ax.set_ylabel("Price")
    ax.set_title("LSTM 1-Step Forecast")
    return fig_to_pil(fig)


# --- Prediction UI ---
def predict(open_p, high_p, low_p, volume, prev_close, day_wk, month):
    if lin_model is None:
        return "Model not loaded."
    X = pd.DataFrame(
        [
            {
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Volume": volume,
                "Prev_Close": prev_close,
                "Day_of_Week": day_wk,
                "Month": month,
            }
        ]
    )
    pred = lin_model.predict(X)[0]
    return f"📈 ₹{pred:.2f}"


# --- Gradio UI ---
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("📊 Trend & Volume"):
            gr.Image(plot_trend_volume)
        with gr.TabItem("💰 Dividends & Splits"):
            gr.Image(plot_div_splits)
        with gr.TabItem("📈 MA Crossover"):
            gr.Image(plot_ma_crossover)
        with gr.TabItem("📉 Daily % Change"):
            gr.Image(plot_daily_change)
        with gr.TabItem("🤖 Linear Model Accuracy"):
            gr.Image(plot_actual_predicted)
        with gr.TabItem("🧠 LSTM Forecast"):
            gr.Image(forecast_lstm)
        with gr.TabItem("🔮 Predict Close Price"):
            open_price = gr.Number(label="Open Price (₹)")
            high_price = gr.Number(label="High Price (₹)")
            low_price = gr.Number(label="Low Price (₹)")
            volume = gr.Number(label="Volume")
            prev_close = gr.Number(label="Previous Close (₹)")
            day_of_week = gr.Number(label="Day of Week (0=Mon)")
            month = gr.Number(label="Month (1–12)")
            output = gr.Textbox(label="Predicted Close Price")
            btn = gr.Button("🔮 Predict")
            btn.click(
                fn=predict,
                inputs=[open_price, high_price, low_price,
                        volume, prev_close, day_of_week, month],
                outputs=output
            )


# Launch App
if __name__ == "__main__":
    demo.launch()

# demo.launch(share=True, server_name="", server_port=7860)
