# ==========================================
# Simple Linear Regression - Laptop Dataset
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
# Using the exact name from your project folder
df = pd.read_csv("laptop_price - dataset.csv")

# Strip hidden spaces from column names to avoid KeyErrors
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Define Feature and Target
# ------------------------------------------
# Selecting only RAM to see its direct impact on Price
try:
    X = df[["RAM (GB)"]].copy()
    y = df["Price (Euro)"].copy()
except KeyError:
    print("Column 'RAM (GB)' not found. Check if the name matches your CSV.")
    # Fallback to the 9th column (index 8) if the name fails
    X = df.iloc[:, [8]].copy()
    y = df.iloc[:, 14].copy()

# Ensure numeric types (removes 'GB' or 'Euro' if they are strings)
X.iloc[:, 0] = pd.to_numeric(X.iloc[:, 0], errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Remove missing values
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
# Step 4: Train Model
# ------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Step 5: Model Parameters
# ------------------------------------------
# The Slope tells you the price increase per 1GB of RAM
print(f"Slope (Price per 1GB): {model.coef_[0]:.2f} Euro")
print(f"Intercept (Base Price): {model.intercept_:.2f} Euro")

# ------------------------------------------
# Step 6: Evaluation
# ------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f} Euro")
print(f"R2 Score: {r2:.4f}")

# ------------------------------------------
# Step 7: Visualization
# ------------------------------------------
plt.figure(figsize=(10, 6))

# Plotting the actual points
sns.scatterplot(x=X.iloc[:, 0], y=y, color='blue', alpha=0.5, label='Actual Laptops')

# Plotting the red regression line
# We use X_train to show the line because it's what the model learned
plt.plot(X_train, model.predict(X_train), color='red', linewidth=3, label='Price Trend Line')

plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.title("Simple Linear Regression: How RAM Affects Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ------------------------------------------
# Step 8: Predict New Laptop Price
# ------------------------------------------
new_ram = np.array([[16]]) # Predict price for 16GB RAM
predicted_price = model.predict(new_ram)

print(f"\nPredicted Price for a 16GB RAM Laptop: {predicted_price[0]:.2f} Euro")