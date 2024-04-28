from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import FictionalRegressor # it is example, not any real algo

# Generate some fictional data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the FictionalRegressor model
model = FictionalRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
score = model.score(X_test, y_test)

print("Model score:", score)