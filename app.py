# app.py
import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = "model/TCS_Stock_Predictor.pkl"

# Load Model Safely
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")


def predict(
    open_price, high_price, low_price, volume, prev_close, day_of_week, month
):
    if model is None:
        return "Model not loaded. Please check the server logs."

    try:
        data = pd.DataFrame(
            [
                [
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(volume),
                    float(prev_close),
                    int(day_of_week),
                    int(month),
                ]
            ],
            columns=[
                "Open",
                "High",
                "Low",
                "Volume",
                "Prev_Close",
                "Day_of_Week",
                "Month",
            ],
        )

        prediction = model.predict(data.values)
        return f"📈 Predicted Close Price: ₹{prediction[0]:.2f}"
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Open Price (₹)"),
        gr.Number(label="High Price (₹)"),
        gr.Number(label="Low Price (₹)"),
        gr.Number(label="Volume"),
        gr.Number(label="Previous Day Close Price (₹)"),
        gr.Number(label="Day of Week (0=Monday, 6=Sunday)"),
        gr.Number(label="Month (1-12)"),
    ],
    outputs=gr.Textbox(label="Predicted Close Price"),
    title="TCS Stock Close Price Predictor - Upgraded",
    description="""Enter Open, High, Low, Volume,
    Previous Close, Day of Week,
    and Month to predict the TCS stock closing price.",
    theme="default""",
)


if __name__ == "__main__":
    iface.launch(server_name="", server_port=7860)
