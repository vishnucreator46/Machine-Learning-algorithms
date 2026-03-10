# DBSCAN Clustering - Laptop Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --- Step 1: Load and Clean Dataset ---
try:
    df = pd.read_csv("laptop_price - dataset.csv")
    
    # Strip whitespace from column names to prevent KeyErrors
    df.columns = df.columns.str.strip()
    print("Columns found in CSV:", df.columns.tolist())
except FileNotFoundError:
    print("Error: The CSV file was not found. Check the file name!")
    exit()

# --- Step 2: Extract & Clean Features ---
# We use the exact names from your screenshot. 
# Note: 'Weight (kg' often lacks the closing ')' in this specific dataset.
try:
    # We create a copy of the columns we need
    features = df[[
        "Inches",
        "CPU_Freq",
        "RAM (GB)",
        "Weight (kg", # Based on your screenshot's column O header
        "Price (Euro)"
    ]].copy()
except KeyError:
    # Fallback if the names above still don't match exactly
    print("Naming mismatch detected. Using column positions instead...")
    # Inches(3), CPU_Freq(7), RAM(8), Weight(13), Price(14)
    features = df.iloc[:, [3, 7, 8, 13, 14]].copy()

# Ensure all data in these columns is treated as a number
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

# Remove any rows with missing values
features = features.dropna()

# --- Step 3: Standardize Data ---
# This scales everything so that Price (thousands) doesn't drown out Inches (teens)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- Step 4: Apply DBSCAN ---
# eps: The distance to look for neighbors. 
# min_samples: Minimum points to form a cluster.
dbscan = DBSCAN(eps=1.0, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)

# Add cluster column back to a copy of the dataframe
df_result = df.loc[features.index].copy()
df_result["Cluster"] = clusters

# --- Step 5: Visualization ---
plt.figure(figsize=(10, 6))

# Plotting RAM vs Price as requested
sns.scatterplot(
    data=df_result,
    x="RAM (GB)",
    y="Price (Euro)",
    hue="Cluster",
    palette="viridis",
    s=100,
    alpha=0.7
)

plt.title("DBSCAN Clustering: RAM vs Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (Euro)")
plt.legend(title="Cluster (-1 = Noise)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- Step 6: Print Results ---
print("\nCluster Assignments (First 10 rows):")
print(df_result[["Company", "TypeName", "RAM (GB)", "Price (Euro)", "Cluster"]].head(10))

# Summary of clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"\nEstimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")