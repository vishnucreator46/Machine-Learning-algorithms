# ==========================================
# Simple Linear Regression - Laptop Dataset
# ==========================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------

df = pd.read_csv("laptop_dataset.csv")

print("First 5 rows:")
print(df.head())

# ------------------------------------------
# Step 3: Define Feature and Target
# ------------------------------------------
# Using only RAM as independent variable

X = df[["RAM (GB)"]]        # Independent variable
y = df["Price (Euro)"]      # Dependent variable

# Remove missing values
X = X.dropna()
y = y.loc[X.index]

# ------------------------------------------
# Step 4: Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# Step 5: Train Model
# ------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Step 6: Model Parameters
# ------------------------------------------

print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# ------------------------------------------
# Step 7: Predictions
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# Step 8: Evaluation
# ------------------------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# ------------------------------------------
# Step 9: Visualization
# ------------------------------------------

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.title("Simple Linear Regression (RAM vs Price)")
plt.legend()
plt.show()

# ------------------------------------------
# Step 10: Predict New Laptop Price
# ------------------------------------------

new_ram = np.array([[16]])   # 16GB RAM
predicted_price = model.predict(new_ram)

print("\nPredicted Price for 16GB RAM Laptop:", predicted_price[0])
