import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
from PIL import Image

# Load model
MODEL_PATH = "model/TCS_Stock_Predictor.pkl"
DATA_PATH = "data/TCS_stock_history.csv"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Model load error: {e}")


# Prediction function
def predict(open_price,
            high_price, low_price, volume, prev_close, day_of_week, month):
    if model is None:
        return "Model not loaded. Please check server logs."
    try:
        data = pd.DataFrame([{
            "Open": float(open_price),
            "High": float(high_price),
            "Low": float(low_price),
            "Volume": float(volume),
            "Prev_Close": float(prev_close),
            "Day_of_Week": int(day_of_week),
            "Month": int(month),
        }])
        prediction = model.predict(data)
        return f"📈 Predicted Close Price: ₹{prediction[0]:.2f}"
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"


# Visualization function returning PIL image
def show_visualizations():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()

    _, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].plot(df["Date"], df["Close"], label="Close Price", color="blue")
    axs[0].plot(df["Date"], df["MA50"], label="MA 50", color="orange")
    axs[0].plot(df["Date"], df["MA200"], label="MA 200", color="green")
    axs[0].set_title("TCS Stock Price & Moving Averages")
    axs[0].legend()
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Price")

    axs[1].plot(df["Date"], df["Volume"], label="Volume", color="purple")
    axs[1].set_title("TCS Trading Volume Over Time")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Volume")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    plt.close()
    return image


# Gradio App UI
with gr.Blocks() as demo:
    with gr.Tab("📊 Visual Insights"):
        gr.Markdown("### 📈 TCS Stock Trends with Moving Averages and Volume")
        vis_output = gr.Image(label="Generated Insights")
        vis_btn = gr.Button("📊 Generate Visuals")
        vis_btn.click(fn=show_visualizations, inputs=[], outputs=vis_output)

    with gr.Tab("🤖 Predict Close Price"):
        with gr.Row():
            open_price = gr.Number(label="Open Price (₹)")
            high_price = gr.Number(label="High Price (₹)")
            low_price = gr.Number(label="Low Price (₹)")
            volume = gr.Number(label="Volume")
            prev_close = gr.Number(label="Previous Day Close Price (₹)")
            day_of_week = gr.Number(label="Day of Week (0=Monday)")
            month = gr.Number(label="Month (1-12)")
        output = gr.Textbox(label="Predicted Close Price")
        btn = gr.Button("Predict")
        btn.click(
            predict,
            inputs=[
                open_price,
                high_price,
                low_price, volume, prev_close, day_of_week, month
                ],
            outputs=output,
        )

if __name__ == "__main__":
    demo.launch()
