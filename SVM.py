# Import Libraries
import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# --- FIXED LOAD DATASET SECTION ---
# This finds the folder where SVM.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# This joins that folder path with your filename
file_path = os.path.join(base_path, "laptop_price - dataset.csv")

try:
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please ensure the CSV file is in the same folder as this script.")
    exit() # Stops the script if the file is missing
# ----------------------------------

# Drop unnecessary column if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Convert Price into Categories
def price_category(price):
    if price < 500:
        return 0 # "Cheap"
    elif price < 1200:
        return 1 # "Medium"
    else:
        return 2 # "Expensive"

# Note: Using numeric categories helps SVM and Naive Bayes perform better 
# than raw strings, though LabelEncoder would handle it too.
df["Price_Category"] = df["Price (Euro)"].apply(price_category)

# Remove original price column
df = df.drop("Price (Euro)", axis=1)

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split Features and Target
X = df.drop("Price_Category", axis=1)
y = df["Price_Category"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
