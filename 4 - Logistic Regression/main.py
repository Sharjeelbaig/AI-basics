import numpy
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the data
data = pd.read_csv('dataset.csv')
X = data.drop('Cancerous', axis=1).values
y = data['Cancerous'].values

logr = LogisticRegression()
logr.fit(X,y)
value = float(input('Enter the tumor size: '))
predicted = logr.predict(numpy.array([value]).reshape(-1,1))

if predicted == 1:
    print('Cancerous')
else:
    print('non-Cancerous')
