# ==========================================
# Multiple Linear Regression - Laptop Dataset
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
# Using the exact name from your project folder
df = pd.read_csv("laptop_price - dataset.csv")

# Strip hidden spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Select Features (Independent Variables)
# ------------------------------------------
try:
    # Matching the specific names from your spreadsheet screenshot
    X = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg"  # Matching the missing ')' from your screenshot
    ]].copy()
except KeyError:
    print("Column mismatch. Using position-based selection instead...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13)
    X = df.iloc[:, [3, 7, 8, 13]].copy()

y = df["Price (Euro)"].copy()

# Ensure all data is numeric (converts any 'kg' or 'GB' strings to floats)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Remove missing values
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
# Step 4: Create and Train Model
# ------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Step 5: Make Predictions
# ------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------
# Step 6: Evaluation
# ------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Euro")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Euro")
print(f"R2 Score (Accuracy): {r2:.4f}")

# ------------------------------------------
# Step 7: Model Parameters (The "Why")
# ------------------------------------------
print(f"\nBase Price (Intercept): {model.intercept_:.2f} Euro")

print("\nPrice increase per unit (Coefficients):")
for feature, coef in zip(X.columns, model.coef_):
    print(f" - {feature}: {coef:.2f} Euro")

# ------------------------------------------
# Step 8: Visualization
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Price (Euro)")
plt.ylabel("Predicted Price (Euro)")
plt.title("Linear Regression: Actual vs. Predicted Prices")
plt.show()

# ------------------------------------------
# Step 9: Predict for New Laptop
# ------------------------------------------
# Inches, CPU_Freq, RAM, Weight
new_specs = [[15.6, 2.5, 8, 2.2]]
predicted_price = model.predict(new_specs)

print(f"\nPredicted Price for new laptop: {predicted_price[0]:.2f} Euro")