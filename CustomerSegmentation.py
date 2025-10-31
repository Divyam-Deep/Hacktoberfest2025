import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample dataset: customers with spending behavior
data = {
    'Age': [25, 34, 22, 45, 52, 23, 40, 60, 48, 30],
    'Annual_Income': [35, 58, 20, 80, 90, 25, 60, 100, 85, 40],
    'Spending_Score': [75, 40, 90, 35, 20, 80, 30, 15, 25, 65]
}
df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize with PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
plt.scatter(components[:,0], components[:,1], c=df['Cluster'], cmap='viridis', s=100)
plt.title("Customer Segments (via K-Means + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

print(df)
