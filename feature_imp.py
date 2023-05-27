import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.inspection import permutation_importance

# Generate random data
X, y = make_blobs(n_samples=1000, centers=4, n_features=10, random_state=42)

# Define KMeans models with and without feature importance sampling
kmeans_no_importance = KMeans(n_clusters=4, random_state=42)
kmeans_with_importance = KMeans(n_clusters=4, random_state=42)

# Fit the model without feature importance sampling
kmeans_no_importance.fit(X)

# Calculate feature importance scores
result = permutation_importance(KMeans(n_clusters=4, random_state=42), X, y, n_repeats=10, random_state=42)
feature_importance_scores = result.importances_mean

# Fit the model with feature importance sampling
kmeans_with_importance.fit(X[:, np.argsort(feature_importance_scores)[-5:]])

# Plot the model accuracy graphs
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('KMeans Clustering Accuracy Comparison')

# Plot the model without feature importance sampling
y_pred_no_importance = kmeans_no_importance.predict(X)
axs[0].scatter(X[:, 0], X[:, 1], c=y_pred_no_importance)
axs[0].set_title('Without Feature Importance Sampling')

# Plot the model with feature importance sampling
y_pred_with_importance = kmeans_with_importance.predict(X[:, np.argsort(feature_importance_scores)[-5:]])
axs[1].scatter(X[:, 0], X[:, 1], c=y_pred_with_importance)
axs[1].set_title('With Feature Importance Sampling')

plt.show()
