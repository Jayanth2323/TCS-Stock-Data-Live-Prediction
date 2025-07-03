---
title: "TCS Stock Live Prediction"
emoji: 📈
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: true
license: mit
python_version: "3.10"
---
## 📈 TCS Stock Data – Live Price Forecasting with ML & DL

An end-to-end interactive stock prediction dashboard for Tata Consultancy Services (TCS) using Linear Regression, LSTM, SHAP explainability, and interactive Plotly analytics.

## 🚀 Features

- 🔮 Predict TCS stock closing price using Linear Regression
- 📉 LSTM-based price forecasting (last 60-day window)
- 📊 Interactive Plotly charts (MAs, Dividends, Volume, Daily Change, and more)
- 📎 SHAP Explainability for prediction transparency
- 🧾 Export detailed PDF reports (Matplotlib fallback)
- 🌐 Gradio Web UI for intuitive interaction
- ✅ Fully containerized with Hugging Face Spaces deployment-ready setup

## 🧠 ML Models

- **Linear Regression**: Trained on historical Open, High, Low, Volume, Prev_Close, Day_of_Week, Month
- **LSTM Model**: Trained on scaled 60-day close prices using TensorFlow

## 📦 Project Structure

├── app.py # Gradio app with SHAP, PDF, Plotly
├── model/
│ ├── TCS_Stock_Predictor.pkl
│ ├── tcs_lstm_model.keras
│ └── tcs_lstm_scaler.pkl
├── data/
│ └── TCS_stock_history.csv
├── predictions/
│ └── tcs_stock_analysis.pdf
└── requirements.txt

## 📸 Preview

![Demo](assets/demo.gif)

## 📥 Run Locally

```bash
git clone https://github.com/Jayanth2323/TCS-Stock-Data-Live-Prediction
cd TCS-Stock-Data-Live-Prediction
pip install -r requirements.txt
python app.py
---
