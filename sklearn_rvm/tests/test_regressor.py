import pytest
import numpy as np
from sklearn_rvm import EMRVR
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn import datasets

boston = datasets.load_boston()
rng = np.random.RandomState(0)
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

def test_simple_fit_predict():
    X = np.array([[0, 0], [2, 2]])
    y = np.array([0.0, 2.5 ])
    clf = EMRVR()
    X_test = np.array([[5,5]])
    clf.fit(X,y)
    pred = clf.predict(X_test)
    assert pred != None

def test_precomputed_fit_predict():
    kernel = pairwise_kernels(boston.data, metric='linear')
    clf = EMRVR(kernel = "precomputed")
    clf.fit(kernel, boston.target)
    pred = clf.predict(kernel)
    assert pred.shape == boston.target.shape
