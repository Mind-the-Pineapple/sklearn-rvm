#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
"""
print(__doc__)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn_rvm import EMRVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

K = pairwise_kernels(X_train)
clf = EMRVC(kernel='precomputed')
clf.fit(K, y_train)

K_test = pairwise_kernels(X_test, X_train)
print(clf.predict(K_test))
print(clf.predict_proba(K_test))
print(clf.score(K_test, y_test))
