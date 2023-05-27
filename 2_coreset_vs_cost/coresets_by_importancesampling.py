from sklearn.datasets import load_iris
import numpy as np

# Load the iris dataset
iris = load_iris()
X = iris.data  # Data points

# Define the number of coresets and their sizes
num_coresets = 3
coreset_sizes = [50, 75, 100,150]  # Sizes of the coreset subsets

# Perform importance sampling for each coreset
coresets = []
for coreset_size in coreset_sizes:
    # Define the importance sampling weights
    weights = np.ones(len(X)) / len(X)  # Equal weight for each data point initially

    # Perform importance sampling
    sample_indices = np.random.choice(len(X), size=coreset_size, replace=True, p=weights)

    # Subsampled dataset based on importance sampling
    coreset = X[sample_indices]

    # Add the coreset to the list of coresets
    coresets.append(coreset)

# Print the sizes of the generated coresets
for i, coreset in enumerate(coresets):
    print(f"Coreset {i+1} size: {len(coreset)}")
