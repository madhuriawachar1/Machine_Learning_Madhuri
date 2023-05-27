import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load iris dataset
iris = load_iris()
X, y = iris['data'], iris['target']
#print(X[:5])

# K-means clustering without importance sampling
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

# Calculate accuracy without sampling
accuracy_no_sampling = np.sum(labels == y) / len(y)
print("Accuracy without sampling: {:.2f}%".format(accuracy_no_sampling * 100))
# K-means clustering with importance sampling
np.random.seed(0)
n_samples = 50
weights = np.exp(-np.sum((X - np.mean(X, axis=0))**2, axis=1))
weights /= np.sum(weights)
indices = np.random.choice(len(X), size=n_samples, replace=False, p=weights)
X_sampled = X[indices]
#print(X_sampled[:5])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_sampled)
labels = kmeans.predict(X)

# Calculate accuracy with sampling
accuracy_with_sampling = np.sum(labels == y) / len(y)
print("Accuracy with row sampling: {:.2f}%".format(accuracy_with_sampling * 100))
"########################col sampling############################################################"


# Perform feature sampling
np.random.seed(0)
feature_mask = np.random.choice([True, False], size=iris.data.shape[1], p=[0.5, 0.5])
while not any(feature_mask):  # Check if there is at least one True value in feature_mask
    feature_mask = np.random.choice([True, False], size=iris.data.shape[1], p=[0.5, 0.5])
X_sampled = iris.data[:, feature_mask]
print(X_sampled[:5])
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sampled, iris.target, test_size=0.2, random_state=0)

# Train K-Means model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_train)

# Predict clusters for testing set
y_pred = kmeans.predict(X_test)

# Calculate accuracy
accuracy_col = accuracy_score(y_test, y_pred)
print("Accuracy with col sampling: {:.2f}%".format(accuracy_col * 100))

# Plot accuracy results
plt.bar(['No sampling', 'With sampling','with col sampling'], [accuracy_no_sampling, accuracy_with_sampling,accuracy_col])
plt.ylim(0, 1)
plt.title('Accuracy comparison')
plt.show()
