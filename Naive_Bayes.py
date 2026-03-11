# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("laptop_price - dataset.csv")

# Drop unnecessary column if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Convert Price into Categories
def price_category(price):
    if price < 500:
        return "Cheap"
    elif price < 1200:
        return "Medium"
    else:
        return "Expensive"

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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes Model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict test data
y_pred = nb_model.predict(X_test)

# Accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))