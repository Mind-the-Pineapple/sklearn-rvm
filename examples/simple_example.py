import numpy as np
from sklearn_rvm import EMRVC

# General a toy dataset:s it's just a straight line with some Gaussian noise:
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]

# Fit the classifier
clf = EMRVC(kernel='linear')
clf.fit(X, y)

X_test = np.linspace(-5, 10, 300)
clf.predict(X)
