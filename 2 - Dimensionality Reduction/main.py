#importing libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../programming language trend over time.csv') 
X = data.iloc[:, 1:].values  # Select all rows and columns from index 1 onwards for features
y = data.iloc[:, 0].values  # Select all rows for the target variable


# Apply PCA
pca = PCA(n_components=2)  # Specify the number of components
transformed_data = pca.fit_transform(X)  # Fit and transform the data

# dimensions of the original data
print('Original data shape:', X.shape)
# dimensions of the transformed data
print('Transformed data shape:', transformed_data.shape)