# app.py
import gradio as gr
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model/TCS_Stock_Predictor.pkl")

def predict(open, high, low):
    data = pd.DataFrame([[open, high, low]], columns=['Open', 'High', 'Low'])
    prediction = model.predict(data)
    return f"Predicted Close Price: {prediction[0]:.2f}"

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="Open"), gr.Number(label="High"), gr.Number(label="Low")],
    outputs="text"
)

if __name__ == "__main__":
    iface.launch()
