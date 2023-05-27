from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(42)
# Load the iris dataset
iris = load_iris()
X = iris.data  # Data points
weights = np.ones(len(X)) / len(X)  # Equal weight for each data point initially

# Define the number of clusters
k = 3

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Calculate the cost
cost = 0
for i, x in enumerate(X):
    #print('wt',weights[i])
    cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
    squared_distance = np.linalg.norm(x - cluster_center) ** 2
    cost += weights[i] * squared_distance
print('cost ')
print("Cost of (X, Q):", cost)


cost_original_ds=cost

# Define the coreset sizes
coreset_sizes = [50, 75, 100,150]  # Sizes of the coreset subsets

# Calculate the cost for each coreset
costs = []
for coreset_size in coreset_sizes:
    # Perform importance sampling
    sample_indices = np.random.choice(len(X), size=coreset_size, replace=True, p=weights)
    coreset = X[sample_indices]

    # Perform k-means clustering on the coreset
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(coreset)

    # Calculate the cost of the coreset centers on the full dataset
    cost = 0
    for x in X:
        closest_center = kmeans.cluster_centers_[kmeans.predict([x])]
        squared_distance = np.linalg.norm(x - closest_center) ** 2
        cost += weights[np.where(X == x)[0][0]] * squared_distance

    costs.append(cost)

# Print the costs
for i, cost in enumerate(costs):
    print(f"Cost of coreset {i+1} centers with respect to the original dataset: {cost}")
    
rel_cost=[element / cost_original_ds for element in costs]   
# Plot the costs vs coreset sizes
plt.plot(coreset_sizes, rel_cost, 'bo-')
plt.xlabel('Coreset Size')
plt.ylabel('Cost')
plt.title('Cost vs Coreset Size')
plt.show()

rel_size=[element / 150 for element in coreset_sizes]   
# Plot the costs vs coreset sizes
plt.plot(rel_size, rel_cost, 'bo-')
plt.xlabel('Coreset Size')
plt.ylabel('Cost')
plt.title('Cost vs Coreset relative Size')
plt.show()
