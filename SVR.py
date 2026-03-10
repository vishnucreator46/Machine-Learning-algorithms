# ==========================================
# Support Vector Regression (SVR)
# Laptop Price Prediction
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
df = pd.read_csv("laptop_price - dataset.csv")

# Remove invisible spaces from headers
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Select Features and Target
# ------------------------------------------
try:
    # Using the exact names from your spreadsheet screenshot
    X = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg"  # Matching the missing ')' in your CSV screenshot
    ]].copy()
except KeyError:
    print("Column mismatch. Using position-based fallback...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13)
    X = df.iloc[:, [3, 7, 8, 13]].copy()

y = df["Price (Euro)"].copy()

# Ensure all data is numeric (cleaning any text like 'kg' or 'GB')
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with missing values
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# ------------------------------------------
# Step 3: Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------------------
# Step 4: Standardize Data (MANDATORY for SVR)
# ------------------------------------------
# SVR uses distance calculations; without scaling, Price would break the model.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# Step 5: Train SVR Model
# ------------------------------------------
# 'rbf' kernel allows SVR to handle non-linear relationships (curves in data)
svr_model = SVR(kernel='rbf', C=1000, epsilon=0.1) 
svr_model.fit(X_train_scaled, y_train)

# ------------------------------------------
# Step 6: Predictions
# ------------------------------------------
y_pred = svr_model.predict(X_test_scaled)

# ------------------------------------------
# Step 7: Evaluation
# ------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- SVR Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f} Euro")
print(f"R2 Score: {r2:.4f}")

# ------------------------------------------
# Step 8: Visualization
# ------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel("Actual Price (Euro)")
plt.ylabel("Predicted Price (Euro)")
plt.title("SVR: Actual vs. Predicted Prices")
plt.legend()
plt.show()

# ------------------------------------------
# Step 9: Predict New Laptop
# ------------------------------------------
# New laptop: 15.6 inch, 2.8 GHz, 16GB RAM, 1.8kg Weight
new_laptop = [[15.6, 2.8, 16, 1.8]]
new_laptop_scaled = scaler.transform(new_laptop)

predicted_price = svr_model.predict(new_laptop_scaled)
print(f"\nPredicted Price for Custom Laptop: {predicted_price[0]:.2f} Euro")