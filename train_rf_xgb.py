# train_rf_xgb.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Paths
DATA_PATH = "data/TCS_stock_history.csv"
RF_MODEL_PATH = "model/rf_model.pkl"
XGB_MODEL_PATH = "model/xgb_model.pkl"

# Load and preprocess
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Prev_Close'] = df['Close'].shift(1)
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek
df.dropna(inplace=True)

# Features and target
features = [
    'Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']
X = df[features]
y = df['Close']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
joblib.dump(rf, RF_MODEL_PATH)
print(f"✅ RF saved at {RF_MODEL_PATH}")

# Train XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X, y)
joblib.dump(xgb, XGB_MODEL_PATH)
print(f"✅ XGB saved at {XGB_MODEL_PATH}")

# Metrics
for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"📊 {name}: MSE = {mse:.2f}, R² = {r2:.2f}")
