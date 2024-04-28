import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
# Load the data
data = pd.read_csv('./dataset.csv') # simple data set showing study hour along with exam score, so there are only two columns
# Prepare the data
X = data.drop('Exam Score', axis=1) 
y = data['Exam Score']
# Train the model
model = LinearRegression()
model.fit(X, y)
# Predict the exam score for a student who studies for 5 hours
hours = np.array([[5]])
predicted_score = model.predict(hours)
print(f'Predicted exam score for a student who studies for 5 hours: {predicted_score[0]:.2f}')