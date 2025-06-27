import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Constants
DATA_PATH = "data/TCS_stock_history.csv"
SAVE_PATH = "model/tcs_lstm_model.h5"
SCALER_PATH = "model/tcs_lstm_scaler.pkl"
SEQUENCE_LENGTH = 60

def prepare_lstm_data(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Load dataset
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Use only 'Close' prices for LSTM
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Save the scaler
joblib.dump(scaler, SCALER_PATH)

# Create sequences
X, y = prepare_lstm_data(data_scaled, SEQUENCE_LENGTH)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data (e.g., 90% train)
split_idx = int(len(X) * 0.9)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Save the model
model.save(SAVE_PATH)
print(f"✅ LSTM model saved to {SAVE_PATH}")
