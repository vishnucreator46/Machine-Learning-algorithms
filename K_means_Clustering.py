# ==========================================
# K-Means Clustering - Laptop Dataset
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------
# Based on your previous error, using the exact file name
df = pd.read_csv("laptop_price - dataset.csv") 

# Clean hidden spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Select Numerical Features
# ------------------------------------------
# Using the exact names from your dataset screenshot
try:
    features = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg",    # Matching the 'missing bracket' in your screenshot
        "Price (Euro)"
    ]].copy()
except KeyError:
    # Fallback to column positions if names fail
    print("Column names didn't match perfectly. Using index positions...")
    features = df.iloc[:, [3, 7, 8, 13, 14]].copy()

# Ensure data is numeric (strips 'kg' or 'GB' if present)
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

features = features.dropna()

# ------------------------------------------
# Step 3: Standardize Data
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ------------------------------------------
# Step 4: Apply KMeans
# ------------------------------------------
# We'll use 3 clusters as a starting point
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels back to the dataframe
df_result = df.loc[features.index].copy()
df_result["Cluster"] = clusters

# ------------------------------------------
# Step 5: Silhouette Score
# ------------------------------------------
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.4f}")

# ------------------------------------------
# Step 6: Visualization
# ------------------------------------------
plt.figure(figsize=(10, 6))

# Plot the laptop data points
sns.scatterplot(
    data=df_result, 
    x="RAM (GB)", 
    y="Price (Euro)", 
    hue="Cluster", 
    palette="viridis", 
    s=100, 
    alpha=0.6
)

# FIX: To plot centroids, we must inverse-transform them back to original scale
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# Plot the centroids (the 'average' laptop for each group)
plt.scatter(
    centroids_original[:, 2],  # Original scale RAM
    centroids_original[:, 4],  # Original scale Price
    marker='*', 
    s=300, 
    c='red', 
    edgecolor='black',
    label='Centroids'
)

plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.title("K-Means Clustering: Laptop Segments")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Print summary
print("\nAverage Price per Cluster:")
print(df_result.groupby("Cluster")["Price (Euro)"].mean())