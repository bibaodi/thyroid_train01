import numpy as np

# Create a sample NumPy ndarray with boolean values
array = np.array([[True, False, False], [False, True, False], [True, True, False]])

# Use np.where to find the indices of True elements
indices = np.where(array)

# Convert the indices to a list of tuples
true_indices = list(zip(indices[0], indices[1]))
true_indices = [(int(row), int(col)) for row, col in zip(indices[0], indices[1])]

# Print the list of indices
print(true_indices)

