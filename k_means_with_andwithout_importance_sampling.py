import numpy as np
from sklearn.cluster import KMeans

# Generate some example data
X = np.random.normal(loc=[0,0], scale=[1,1], size=(1000, 2))

# Define the number of clusters and importance sampling weights
k = 5
weights = np.random.uniform(low=0, high=1, size=X.shape[0])

# Perform k-means clustering with importance sampling
kmeans_importance = KMeans(n_clusters=k).fit(X, sample_weight=weights)

# Perform k-means clustering without importance sampling
kmeans_no_importance = KMeans(n_clusters=k).fit(X)

# Compare the clustering results
print("Clusters with importance sampling:\n", kmeans_importance.labels_)
print("Clusters without importance sampling:\n", kmeans_no_importance.labels_)
