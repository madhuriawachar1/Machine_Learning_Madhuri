from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np

# Load the iris dataset
iris = load_iris()


X = iris.data  # Data points
n = len(X)  # Number of data points
weights = [(1 / n) for i in range(n)]  # Weight for each data point

# Define the number of clusters
k = 3

# Perform k-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Calculate the cost
cost = 0
for i, x in enumerate(X):
    print('wt',weights[i])
    cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
    squared_distance = np.linalg.norm(x - cluster_center) ** 2
    cost += weights[i] * squared_distance

print("Cost of (X, Q):", cost)
