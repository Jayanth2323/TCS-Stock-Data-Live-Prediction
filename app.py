# app.py (✅ FINAL: SHAP Fixed + Plotly + PDF + Gradio UI)
import gradio as gr
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import shap
import io
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima
import datetime
import tensorflow as tf
from PIL import Image

# --- Force CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Paths ---
LINEAR_MODEL_PATH = "model/TCS_Stock_Predictor.pkl"
LSTM_MODEL_PATH = "model/tcs_lstm_model.keras"
SCALER_PATH = "model/tcs_lstm_scaler.pkl"
DATA_PATH = "data/TCS_stock_history.csv"
PDF_PATH = "predictions/tcs_stock_analysis.pdf"
RF_MODEL_PATH = "model/rf_model.pkl"
XGB_MODEL_PATH = "model/xgb_model.pkl"

rf_model = joblib.load(
    RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
xgb_model = joblib.load(
    XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# --- Load Models ---
lin_model = (
    joblib.load(
        LINEAR_MODEL_PATH) if os.path.exists(LINEAR_MODEL_PATH) else None
)
lstm_model = (
    tf.keras.models.load_model(LSTM_MODEL_PATH)
    if os.path.exists(LSTM_MODEL_PATH)
    else None
)
scaler = (
    joblib.load(
        SCALER_PATH) if os.path.exists(SCALER_PATH) else None
)
rf_model = (
    joblib.load(
        RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
)
xgb_model = (
    joblib.load(
        XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
)


# --- Load Data ---
def load_df():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")


# --- SHAP Plot ---
def get_shap_plot(input_data):
    if lin_model is None:
        raise ValueError("Linear model not loaded.")

    df = load_df()
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Prev_Close",
        "Day_of_Week",
        "Month",
    ]
    X_base = df[feature_cols]
    input_df = pd.DataFrame([input_data], columns=feature_cols)

    explainer = shap.Explainer(
        lin_model.predict, X_base, feature_names=feature_cols)
    shap_values = explainer(input_df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# --- Plotly Visuals ---
def plot_combined():
    df = load_df()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ShortMA"] = df["Close"].rolling(20).mean()
    df["LongMA"] = df["Close"].rolling(50).mean()
    df["DailyChange"] = df["Close"].pct_change() * 100
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)

    feats = [
        "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week", "Month"]
    y_true = df["Close"]
    y_pred = lin_model.predict(df[feats]) if lin_model else [0] * len(df)
    mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

    # Random Forest
    y_rf = rf_model.predict(df[feats]) if rf_model else [0] * len(df)
    mse_rf, r2_rf = mean_squared_error(y_true, y_rf), r2_score(y_true, y_rf)

    # XGBoost
    y_xgb = xgb_model.predict(df[feats]) if xgb_model else [0] * len(df)
    mse_xgb, r2_xgb = mean_squared_error(
        y_true, y_xgb), r2_score(y_true, y_xgb)

    lstm_pred = 0
    if lstm_model and scaler:
        seq = scaler.transform(df[["Close"]])[-60:].reshape(1, 60, 1)
        lstm_pred = lstm_model.predict(seq)[0, 0]

    return [
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=df[c], name=c)
                for c in ["Close", "MA50", "MA200"]
            ]
        ).update_layout(title="Close Price & MAs"),
        go.Figure(
            [go.Scatter(x=df["Date"], y=df["Volume"], name="Volume")]
        ).update_layout(title="Trading Volume"),
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=df.get(c, 0), name=c)
                for c in ["Dividends", "Stock Splits"]
            ]
        ).update_layout(title="Dividends & Splits"),
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=df[c], name=c)
                for c in ["Close", "ShortMA", "LongMA"]
            ]
        ).update_layout(title="MA Crossover"),
        go.Figure(
            go.Histogram(
                x=df["DailyChange"].dropna(), nbinsx=50)).update_layout(
            title="Daily % Change"
        ),
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=y_true, name="Actual"),
                go.Scatter(x=df["Date"], y=y_pred, name="Predicted"),
            ]
        ).update_layout(title=f"Linear Model (MSE={mse:.2f}, R²={r2:.2f})"),
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=y_true, name="Actual"),
                go.Scatter(x=df["Date"], y=y_rf, name="Random Forest")
            ]
        ).update_layout(
            title=f"Random Forest (MSE={mse_rf:.2f}, R²={r2_rf:.2f})"),
        go.Figure(
            [
                go.Scatter(x=df["Date"], y=y_true, name="Actual"),
                go.Scatter(x=df["Date"], y=y_xgb, name="XGBoost")
            ]
        ).update_layout(title=f"XGBoost (MSE={mse_xgb:.2f}, R²={r2_xgb:.2f})"),
        go.Figure(
            go.Bar(
                x=["LSTM Forecast"],
                y=[lstm_pred],
                marker=dict(color="orange")
            )
        ).update_layout(title="LSTM Forecast"),
    ]


def forecast_arima(n_days=30):
    df = load_df()
    df = df.set_index("Date")
    df = df.asfreq("D").fillna(method="ffill")

    model = auto_arima(df["Close"], seasonal=False, suppress_warnings=True)
    future = model.predict(n_periods=n_days)

    last_date = df.index[-1]
    future_dates = [
        last_date + datetime.timedelta(days=i+1) for i in range(n_days)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=future, name="ARIMA Forecast",
            line=dict(color="green"))
    )
    fig.update_layout(title=f"ARIMA Forecast - Next {n_days} Days")
    return fig


# --- Export PDF ---
def export_combined_pdf():
    df = load_df()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ShortMA"] = df["Close"].rolling(20).mean()
    df["LongMA"] = df["Close"].rolling(50).mean()
    df["DailyChange"] = df["Close"].pct_change() * 100
    df["Prev_Close"] = df["Close"].shift(1)
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df.dropna(inplace=True)
    X = df[[
        "Open", "High", "Low", "Volume", "Prev_Close", "Day_of_Week", "Month"]]
    y = df["Close"]
    y_pred = lin_model.predict(X)
    y_rf = rf_model.predict(X) if rf_model else [0] * len(X)
    y_xgb = xgb_model.predict(X) if xgb_model else [0] * len(X)
    mse, r2 = mean_squared_error(y, y_pred), r2_score(y, y_pred)
    # Removed unused mse_rf, r2_rf, mse_xbg, r2_xbg calculation

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    axs[0, 0].set_title("Close & MA")
    axs[0, 1].plot(df["Date"], df["Volume"])
    axs[0, 1].set_title("Volume")
    axs[1, 0].set_title("Dividends & Splits")
    axs[1, 1].hist(df["DailyChange"].dropna(), bins=50)
    axs[1, 1].set_title("Daily % Change")
    axs[2, 0].plot(df["Date"], y)
    axs[2, 0].plot(df["Date"], y_pred)
    axs[2, 0].set_title(f"Linear Model (MSE={mse:.2f}, R²={r2:.2f})")
    lstm_val = (
        lstm_model.predict(
            scaler.transform(df[["Close"]])[-60:].reshape(1, 60, 1))[0, 0]
        if lstm_model and scaler
        else 0
    )

    axs[0, 0].plot(df["Date"], df["Close"], label="Actual")
    axs[0, 0].plot(df["Date"], df["MA50"], label="MA50")
    axs[0, 0].legend()

    axs[1, 0].plot(df["Date"], df.get("Dividends", 0), label="Dividends")
    axs[1, 0].plot(df["Date"], df.get("Stock Splits", 0), label="Splits")
    axs[1, 0].legend()

    axs[1, 1].hist(df["DailyChange"].dropna(), bins=50)
    axs[1, 1].set_title("Daily % Change")

    axs[2, 0].plot(df["Date"], y, label="Actual")
    axs[2, 0].plot(df["Date"], y_pred, label="Linear")
    axs[2, 0].plot(df["Date"], y_rf, label="RF")
    axs[2, 0].plot(df["Date"], y_xgb, label="XGB")
    axs[2, 0].set_title("Model Predictions")
    axs[2, 0].legend()

    axs[2, 1].bar(["LSTM Forecast"], [lstm_val], color="orange")
    axs[2, 1].set_title("LSTM Forecast")
    plt.tight_layout()
    fig.savefig(PDF_PATH, format="pdf")
    return PDF_PATH


# --- Prediction ---
def predict(open_p, high_p, low_p, volume, prev_close, day_wk, month):
    if not lin_model:
        return "Model not loaded.", None
    input_dict = {
        "Open": open_p,
        "High": high_p,
        "Low": low_p,
        "Volume": volume,
        "Prev_Close": prev_close,
        "Day_of_Week": day_wk,
        "Month": month,
    }
    X = pd.DataFrame([input_dict])
    pred = lin_model.predict(X)[0]
    shap_img = get_shap_plot(input_dict)
    return f"📈 ₹{pred:.2f}", shap_img


# --- Custom CSS to Fix Bottom Padding ---
custom_css = """
body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #121212;
    overflow-x: hidden;
}
footer {
    display: none !important;
}
.gradio-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    padding-bottom: 0 !important;
}
main {
    flex: 1;
    padding-bottom: 0 !important;
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    with gr.Tabs():
        with gr.TabItem("📊 All-in-One Analysis"):
            for f in plot_combined():
                gr.Plot(value=f)
            btn = gr.Button("⬇️ Download PDF Report")
            pdf_out = gr.File()
            btn.click(fn=export_combined_pdf, outputs=pdf_out)

        with gr.TabItem("📈 ARIMA Forecast"):
            arima_plot = gr.Plot()
            with gr.Row():
                gr.Button("🔮 Forecast 30 Days").click(
                    fn=lambda: forecast_arima(30), outputs=arima_plot
                )
                gr.Button("🔮 Forecast 90 Days").click(
                    fn=lambda: forecast_arima(90), outputs=arima_plot
                )
                gr.Button("🔮 Forecast 180 Days").click(
                    fn=lambda: forecast_arima(180), outputs=arima_plot
                )

        with gr.TabItem("🔮 Predict Close Price"):
            open_p = gr.Number(label="Open ₹")
            high_p = gr.Number(label="High ₹")
            low_p = gr.Number(label="Low ₹")
            volume = gr.Number(label="Volume")
            prev_close = gr.Number(label="Previous Close ₹")
            day_wk = gr.Number(label="Day of Week (0=Mon)")
            month = gr.Number(label="Month")
            output = gr.Textbox(label="Predicted Close Price")
            shap_img = gr.Image(label="SHAP Explainability")
            gr.Button("🔮 Predict").click(
                predict,
                inputs=[
                    open_p, high_p, low_p, volume, prev_close, day_wk, month],
                outputs=[output, shap_img],
            )

if __name__ == "__main__":
    demo.launch()
