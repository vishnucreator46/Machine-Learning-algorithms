# ==========================================
# KNN Regression - Laptop Price Prediction
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
# Using the exact name from your previous prompts
df = pd.read_csv("laptop_price - dataset.csv")

# Clean hidden spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Select Features (X) and Target (y)
# ------------------------------------------
try:
    # Based on your screenshot, using exact names found in the CSV
    X = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg"  # Matching the 'missing bracket' in your screenshot
    ]].copy()
    y = df["Price (Euro)"].copy()
except KeyError:
    print("Column names mismatch. Falling back to position-based selection...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13)
    X = df.iloc[:, [3, 7, 8, 13]].copy()
    y = df.iloc[:, 14].copy() # Price(14)

# Ensure numeric types (removes 'kg' or 'GB' text if they exist)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows where any data is missing
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# ------------------------------------------
# Step 3: Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# Step 4: Standardize Features
# ------------------------------------------
# KNN relies on distance, so scaling is MANDATORY.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# Step 5: Apply KNN Regressor
# ------------------------------------------
# n_neighbors=5 is a good start. 
# Increasing this makes predictions smoother; decreasing it makes them more "jumpy".
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# ------------------------------------------
# Step 6: Predictions
# ------------------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------------------
# Step 7: Evaluation
# ------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} Euro")
print(f"R2 Score (Accuracy): {r2:.4f}")

# ------------------------------------------
# Step 8: Visualization
# ------------------------------------------
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.xlabel("Actual Price (Euro)")
plt.ylabel("Predicted Price (Euro)")
plt.title("KNN Prediction: Actual vs. Predicted Laptop Prices")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------
# Step 9: Make a custom prediction
# ------------------------------------------
# Prediction for: 15.6 inch, 2.5 GHz CPU, 8GB RAM, 2.1kg Weight
sample_laptop = scaler.transform([[15.6, 2.5, 8, 2.1]])
prediction = model.predict(sample_laptop)
print(f"\nPredicted price for a custom laptop: {prediction[0]:.2f} Euro")