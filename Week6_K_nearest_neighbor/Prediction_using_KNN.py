import numpy as np

# Training data
X_train = np.array([
    [-15, -11, 12],
    [-13, 11, 13],
    [-9, 9, 5],
    [1, 10, -6],
    [6, -15, -1],
    [-15, -10, -11]
])
y_train = np.array([-64, -76, -49, -4, 38, -39])

# Test data point
X_test = np.array([1, -10, 7])

# Calculate Manhattan distance (L1) between the test point and each training data point
distances = np.sum(np.abs(X_train - X_test), axis=1)

# Find the K=3 nearest neighbors
K = 1
nearest_neighbor_ids = np.argsort(distances)[:K]

# Compute the mean y of the nearest neighbors
y_pred = np.mean(y_train[nearest_neighbor_ids])
nearest_neighbors_with_distances = list(zip(nearest_neighbor_ids, distances[nearest_neighbor_ids]))
print(f"The nearest neighbors are: {nearest_neighbors_with_distances}")
print(f"The predicted value for the test point is: {y_pred}")
