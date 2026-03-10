# ==========================================
# Logistic Regression - Budget vs. Premium
# ==========================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
# Using the exact name from your previous prompts
df = pd.read_csv("laptop_price - dataset.csv")

# Clean hidden spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Create Target Category (₹50,000 threshold)
# ------------------------------------------
# Convert Euro to Rupees (approx conversion)
df["Price_Rupees"] = df["Price (Euro)"] * 90

# 0 = Budget (False/0), 1 = Premium (True/1)
df["Category"] = (df["Price_Rupees"] >= 50000).astype(int)

# ------------------------------------------
# Step 3: Select Features
# ------------------------------------------
try:
    # Matching the specific names from your spreadsheet screenshot
    X = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg"  # Matching the missing ')'
    ]].copy()
except KeyError:
    print("Column mismatch. Using position-based selection instead...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13)
    X = df.iloc[:, [3, 7, 8, 13]].copy()

y = df["Category"]

# Ensure numeric types and handle missing data
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.dropna()
y = y.loc[X.index]

# ------------------------------------------
# Step 4: Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------------------
# Step 5: Standardization (Crucial for Logistic Regression)
# ------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# Step 6: Train Model
# ------------------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ------------------------------------------
# Step 7: Predict
# ------------------------------------------
y_pred = model.predict(X_test_scaled)

# ------------------------------------------
# Step 8: Evaluate
# ------------------------------------------
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Budget", "Premium"]))

# ------------------------------------------
# Step 9: Visualization (Confusion Matrix)
# ------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=["Budget", "Premium"], 
            yticklabels=["Budget", "Premium"])
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Logistic Regression: Confusion Matrix')
plt.show()