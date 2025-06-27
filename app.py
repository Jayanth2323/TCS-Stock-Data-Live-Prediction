import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import os

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Paths ---
LINEAR_MODEL_PATH = "model/TCS_Stock_Predictor.pkl"
LSTM_MODEL_PATH = "model/tcs_lstm_model.keras"
SCALER_PATH = "model/tcs_lstm_scaler.pkl"
DATA_PATH = "data/TCS_stock_history.csv"

# --- Model Load ---
print("Model Exists?", os.path.exists(LSTM_MODEL_PATH))
print("Scaler Exists?", os.path.exists(SCALER_PATH))

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


# --- Helpers ---
def load_df():
    df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# --- Combined Analytics Plot ---
def plot_combined_analysis():
    df = load_df()

    # Preprocessing
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ShortMA"] = df["Close"].rolling(20).mean()
    df["LongMA"] = df["Close"].rolling(50).mean()
    df["DailyChange"] = df["Close"].pct_change() * 100
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)

    # Linear Model Predictions
    feats = [
        "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week", "Month"]
    X = df[feats]
    y_true = df["Close"]
    y_pred = lin_model.predict(X) if lin_model else [0] * len(X)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # LSTM Forecast
    lstm_pred = 0
    if lstm_model and scaler:
        data = scaler.transform(df[["Close"]])
        seq = data[-60:].reshape(1, 60, 1)
        lstm_pred = lstm_model.predict(seq)[0, 0]

    # Combined Figure
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.tight_layout(pad=4)

    # 1. Trend + MA
    axs[0, 0].plot(df["Date"], df["Close"], label="Close")
    axs[0, 0].plot(df["Date"], df["MA50"], label="MA50")
    axs[0, 0].plot(df["Date"], df["MA200"], label="MA200")
    axs[0, 0].set_title("Close Price & Moving Averages")
    axs[0, 0].legend()

    # 2. Volume
    axs[0, 1].plot(df["Date"], df["Volume"], label="Volume", color="purple")
    axs[0, 1].set_title("Trading Volume")
    axs[0, 1].legend()

    # 3. Dividends & Splits
    axs[1, 0].plot(df["Date"], df.get("Dividends", 0), label="Dividends")
    axs[1, 0].plot(df["Date"], df.get("Stock Splits", 0), label="Stock Splits")
    axs[1, 0].set_title("Dividends and Stock Splits")
    axs[1, 0].legend()

    # 4. MA Crossover
    axs[1, 1].plot(df["Date"], df["Close"], label="Close")
    axs[1, 1].plot(df["Date"], df["ShortMA"], label="20-day MA")
    axs[1, 1].plot(df["Date"], df["LongMA"], label="50-day MA")
    axs[1, 1].set_title("MA Crossover")
    axs[1, 1].legend()

    # 5. Daily % Change Histogram
    axs[2, 0].hist(df["DailyChange"].dropna(), bins=50, color="teal")
    axs[2, 0].set_title("Daily % Change")

    # 6. Actual vs Prediction + LSTM
    axs[2, 1].plot(df["Date"], y_true, label="Actual")
    axs[2, 1].plot(df["Date"], y_pred, label="Linear Pred", alpha=0.7)
    axs[2, 1].bar(
        ["LSTM Next"], [lstm_pred], color="orange", label="LSTM Forecast")
    axs[2, 1].set_title(f"Model Forecasts (MSE={mse:.2f}, R²={r2:.2f})")
    axs[2, 1].legend()

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
        with gr.TabItem("📊 All-in-One Analysis"):
            gr.Image(plot_combined_analysis)
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
                inputs=[
                    open_price,
                    high_price,
                    low_price,
                    volume,
                    prev_close,
                    day_of_week,
                    month,
                ],
                outputs=output,
            )


# --- Launch App ---
if __name__ == "__main__":
    demo.launch()
