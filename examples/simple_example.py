#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Simple example
=========================================================
"""
print(__doc__)

import numpy as np
from sklearn_rvm import EMRVC

# General a toy dataset:s it's just a straight line with some Gaussian noise:
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

print(clf.predict(X))
print(clf.predict_proba(X))
print(clf.score(X, y))
