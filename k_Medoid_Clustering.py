# ==========================================
# K-Medoids Clustering - Laptop Dataset
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Note: You may need to run 'pip install scikit-learn-extra'
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------------------------------
# Step 1: Load and Clean Dataset
# ------------------------------------------
df = pd.read_csv("laptop_price - dataset.csv") 

# Clean hidden spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Step 2: Select Numerical Features
# ------------------------------------------
try:
    # Matching the specific names from your spreadsheet screenshot
    features = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg",    # Missing ')' as seen in your screenshot
        "Price (Euro)"
    ]].copy()
except KeyError:
    print("Column names mismatch. Falling back to position-based selection...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13), Price(14)
    features = df.iloc[:, [3, 7, 8, 13, 14]].copy()

# Ensure numeric types and handle missing data
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

features = features.dropna()

# ------------------------------------------
# Step 3: Standardize Data
# ------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ------------------------------------------
# Step 4: Apply K-Medoids
# ------------------------------------------
# PAM (Partitioning Around Medoids) is the most common algorithm for K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=42, method='pam')
clusters = kmedoids.fit_predict(X_scaled)

# Add cluster column back to a copy of the dataframe
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

# Plot the laptops
sns.scatterplot(
    data=df_result,
    x="RAM (GB)",
    y="Price (Euro)",
    hue="Cluster",
    palette="Set1",
    s=100,
    alpha=0.6
)

# FIX: Inverse transform the medoid centers to plot them on the RAM/Price scale
medoids_scaled = kmedoids.cluster_centers_
medoids_original = scaler.inverse_transform(medoids_scaled)

# Plot the Medoids (the "Representative" laptops for each cluster)
plt.scatter(
    medoids_original[:, 2], # RAM index
    medoids_original[:, 4], # Price index
    marker='*', 
    s=400, 
    c='yellow', 
    edgecolor='black', 
    label='Medoids (Centers)'
)

plt.title("K-Medoids Clustering: RAM vs Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ------------------------------------------
# Step 7: Print Medoids (The actual laptops chosen as centers)
# ------------------------------------------
print("\nThe 'Typical' Laptops for each cluster (Medoids):")
print(df_result.iloc[kmedoids.medoid_indices_][["Company", "TypeName", "RAM (GB)", "Price (Euro)"]])