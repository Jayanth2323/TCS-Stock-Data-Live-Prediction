# model/model_train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('./data/TCS_stock_history.csv')

# Example features and target
X = data[['Open', 'High', 'Low']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

joblib.dump(
    model,
    './model/TCS_Stock_Predictor.pkl',
    protocol=4
)
print("Model saved successfully.")