# app.py (Final – Gradio + Plotly + SHAP + CSV + PDF + HF Ready)
import gradio as gr
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import io, os, joblib, shap, csv
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
import tensorflow as tf

# --- Init & Paths ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
LINEAR_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH = (
    "model/TCS_Stock_Predictor.pkl",
    "model/tcs_lstm_model.keras",
    "model/tcs_lstm_scaler.pkl",
)
DATA_PATH, PDF_PATH, CSV_LOG_PATH = (
    "data/TCS_stock_history.csv",
    "predictions/tcs_report.pdf",
    "predictions/predicted_log.csv",
)
os.makedirs("predictions", exist_ok=True)

# --- Load Models ---
lin_model = (
    joblib.load(LINEAR_MODEL_PATH)
    if os.path.exists(LINEAR_MODEL_PATH)
    else None
)
lstm_model = (
    tf.keras.models.load_model(LSTM_MODEL_PATH)
    if os.path.exists(LSTM_MODEL_PATH)
    else None
)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


# --- Load & Prep Data ---
def load_df():
    df = pd.read_csv(DATA_PATH, encoding="utf-8", on_bad_lines="skip")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")


# --- Plotly Visuals ---
def plot_price_ma():
    df = load_df()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return go.Figure(
        [go.Scatter(
            x=df["Date"], y=df[c], name=c) for c in ["Close", "MA50", "MA200"]]
    ).update_layout(title="Price & Moving Averages")


def plot_volume():
    df = load_df()
    return go.Figure(
        go.Scatter(
            x=df["Date"], y=df["Volume"], name="Volume",
            line=dict(color="purple")
        )
    ).update_layout(title="Trading Volume")


def plot_dividends():
    df = load_df()
    return go.Figure(
        [
            go.Scatter(x=df["Date"], y=df.get(col, 0), name=col)
            for col in ["Dividends", "Stock Splits"]
        ]
    ).update_layout(title="Dividends & Splits")


def plot_ma_crossover():
    df = load_df()
    df["Short"] = df["Close"].rolling(20).mean()
    df["Long"] = df["Close"].rolling(50).mean()
    return go.Figure(
        [go.Scatter(
            x=df["Date"], y=df[c], name=c) for c in ["Close", "Short", "Long"]]
    ).update_layout(title="MA Crossover Strategy")


def plot_daily_change():
    df = load_df()
    df["Change"] = df["Close"].pct_change() * 100
    return go.Figure(
        go.Histogram(x=df["Change"].dropna(), nbinsx=50)).update_layout(
        title="Daily % Change"
    )


def plot_model_accuracy():
    df = load_df()
    df[["Prev_Close"]] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)
    feats, y = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Prev_Close",
        "Day_of_Week",
        "Month",
    ], df["Close"]
    y_pred = lin_model.predict(df[feats]) if lin_model else [0] * len(df)
    title = f"""Linear Model Accuracy (MSE={mean_squared_error(y, y_pred):.2f},
    R²={r2_score(y, y_pred):.2f})"""
    return go.Figure(
        [
            go.Scatter(x=df["Date"], y=y, name="Actual"),
            go.Scatter(x=df["Date"], y=y_pred, name="Predicted"),
        ]
    ).update_layout(title=title)


def plot_lstm_forecast():
    df = load_df()
    yhat = (
        lstm_model.predict(scaler.transform(
            df[["Close"]])[-60:].reshape(1, 60, 1))[
            0, 0
        ]
        if lstm_model and scaler
        else 0
    )
    return go.Figure(
        go.Bar(x=["LSTM Forecast"], y=[yhat], marker_color="orange")
    ).update_layout(title="LSTM 1-Day Forecast")


# --- Matplotlib PDF Export ---
def generate_pdf():
    df = load_df()
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    plt.tight_layout()
    df[["MA50", "MA200"]] = (
        df["Close"].rolling(50).mean(),
        df["Close"].rolling(200).mean(),
    )
    df[["Short", "Long"]] = (
        df["Close"].rolling(20).mean(),
        df["Close"].rolling(50).mean(),
    )
    df["Change"], df["Prev_Close"] = df["Close"].pct_change() * 100,
    df["Close"].shift(
        1
    )
    df["Day_of_Week"], df["Month"] = df["Date"].dt.dayofweek,
    df["Date"].dt.month
    df.dropna(inplace=True)
    X, y = (
        df[[
            "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week",
            "Month"
            ]],
        df["Close"],
    )
    y_pred = lin_model.predict(X),
    mse = mean_squared_error(y, y_pred),
    r2 = r2_score(y, y_pred),
    axs[0, 0].plot(df["Date"], df["Close"])
    axs[0, 0].plot(df["Date"], df["MA50"])
    axs[0, 0].set_title("Close & MA")
    axs[0, 1].plot(df["Date"], df["Volume"])
    axs[0, 1].set_title("Volume")
    axs[1, 0].plot(df["Date"], df.get("Dividends", 0))
    axs[1, 0].plot(df["Date"], df.get("Stock Splits", 0))
    axs[1, 0].set_title("Dividends & Splits")
    axs[1, 1].hist(df["Change"].dropna(), bins=50)
    axs[1, 1].set_title("Daily % Change")
    axs[2, 0].plot(df["Date"], y)
    axs[2, 0].plot(df["Date"], y_pred)
    axs[2, 0].set_title(f"Linear Model (MSE={mse:.2f}, R2={r2:.2f})")
    axs[2, 1].bar(
        ["Forecast"],
        [
            (
                lstm_model.predict(
                    scaler.transform(df[["Close"]])[-60:].reshape(1, 60, 1)
                )[0, 0]
                if lstm_model and scaler
                else 0
            )
        ],
    )
    axs[2, 1].set_title("LSTM Forecast")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    buf.seek(0)
    with open(PDF_PATH, "wb") as f:
        f.write(buf.read())
    return PDF_PATH


# --- CSV Logging + SHAP ---
def log_prediction(data: dict, pred: float):
    data["Predicted_Close"] = round(pred, 2)
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not os.path.exists(CSV_LOG_PATH):
            writer.writeheader()
        writer.writerow(data)


def get_shap_plot(input_dict):
    df = load_df()
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)
    X = df[[
        "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week", "Month"]]
    shap_values = shap.Explainer(lin_model, X)(pd.DataFrame([input_dict]))
    buf = io.BytesIO()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.clf()
    return Image.open(buf)


# --- Gradio UI ---
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("📊 Visual Insights"):
            gr.Markdown(
                """
            **Close Price Trends** + MA50 & MA200 = Long-Term Pattern
            **Volume** = Market Activity Indicator
            **Dividends & Splits** = Corporate Action Timeline
            **MA Crossover** = Strategy Signal (Short vs Long)
            **Daily % Change** = Volatility Measure
            **Linear vs Actual** = Model Accuracy
            **LSTM Forecast** = Tomorrow’s Close Price Guess
            """
            )
            with gr.Row():
                gr.Plot(plot_price_ma)
                gr.Plot(plot_volume)
            with gr.Row():
                gr.Plot(plot_dividends)
                gr.Plot(plot_ma_crossover)
            with gr.Row():
                gr.Plot(plot_daily_change)
                gr.Plot(plot_model_accuracy)
            with gr.Row():
                gr.Plot(plot_lstm_forecast)
            pdf_btn, pdf_out = gr.Button("⬇️ Download PDF Report"), gr.File()
            pdf_btn.click(generate_pdf, outputs=pdf_out)

        with gr.TabItem("🔮 Predict Close Price"):
            with gr.Row():
                open_ = gr.Number(label="Open ₹")
                high_ = gr.Number(label="High ₹")
                low_ = gr.Number(label="Low ₹")
                vol_ = gr.Number(label="Volume")
            with gr.Row():
                prev_ = gr.Number(label="Prev Close ₹")
                day_ = gr.Number(label="Day (0=Mon)")
                mon_ = gr.Number(label="Month")
            predict_btn = gr.Button("Predict")
            out, shap_img, csv_out = gr.Textbox(), gr.Image(), gr.File()

            def predict_model(o, h, L, v, p, d, m):
                X = {
                    "Open": o,
                    "High": h,
                    "Low": L,
                    "Volume": v,
                    "Prev_Close": p,
                    "Day_of_Week": d,
                    "Month": m,
                }
                pred = lin_model.predict(
                    pd.DataFrame([X]))[0] if lin_model else 0
                log_prediction(X, pred)
                return f"📈 ₹{pred:.2f}", get_shap_plot(X)

            predict_btn.click(
                predict_model,
                [open_, high_, low_, vol_, prev_, day_, mon_],
                [out, shap_img],
            )
            gr.Button("⬇️ Download CSV Log").click(
                fn=lambda: CSV_LOG_PATH, outputs=csv_out
            )


if __name__ == "__main__":
    demo.launch()
