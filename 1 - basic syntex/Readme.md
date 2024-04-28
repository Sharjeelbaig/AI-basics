# Syntex

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # although we don't use it for simplicity but for general discipline, we should import it
from sklearn.any_module import any_algorithm

# load the data
data = pd.read_csv('data.csv')

# split the data
X = data.drop('target', axis=1)
y = data['target']

# create the model
model = any_algorithm()
model.fit(X, y)

# do anythin with the model
if model.predict(X[0]) == y[0]:
    print('model is working fine')
else:
    print('model is not working fine')
```

- [Explained notebook](./explained.ipynb)
- [Direct code](./main.py)
