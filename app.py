# app.py
import gradio as gr
import joblib

# import numpy as np
import pandas as pd
# import os

MODEL_PATH = "model/TCS_Stock_Predictor.pkl"

# Load Model Safely
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")


def predict(inputs):
    if model is None:
        return "Model not loaded. Please check the server logs."

    try:
        open_price, high_price, low_price = inputs
        data = pd.DataFrame([[
            float(open_price),
            float(high_price),
            float(low_price)]],
                            columns=["Open", "High", "Low"])
        prediction = model.predict(data)
        return f"📈 Predicted Close Price: ₹{prediction[0]:.2f}"
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Open Price (₹)"),
        gr.Number(label="High Price (₹)"),
        gr.Number(label="Low Price (₹)")
    ],
    outputs=gr.Textbox(label="Predicted Close Price"),
    title="TCS Stock Close Price Predictor",
    description="""Enter Open, High, and Low
    values to predict the closing stock price of TCS.""",
    theme="default"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
