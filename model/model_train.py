# model/model_train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
data = pd.read_csv('./data/TCS_stock_history.csv')

# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Feature Engineering
data['Prev_Close'] = data['Close'].shift(1)
data['Month'] = data['Date'].dt.month
data['Day_of_Week'] = data['Date'].dt.dayofweek

# Drop rows with NaN values (from shift)
data.dropna(inplace=True)

# Features & Target
feature_cols = [
    'Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month'
]
X = data[feature_cols]
y = data['Close']

print(f"✅ Data prepared. Shape: {X.shape}")
print(f"✅ Features used: {feature_cols}")

# Create a pipeline (even if only one step, useful for future expansion)
pipeline = Pipeline([
    ("regressor", LinearRegression())
])

# Train the pipeline
pipeline.fit(X, y)
print("✅ Model training completed.")

# Ensure model directory exists
os.makedirs("./model", exist_ok=True)

# Save the full pipeline
joblib.dump(pipeline, './model/TCS_Stock_Predictor.pkl', protocol=4)
print("✅ Pipeline saved successfully to './model/TCS_Stock_Predictor.pkl'")
