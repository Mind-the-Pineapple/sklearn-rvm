#import pytest
import numpy as np
from sklearn_rvm import EMRVC
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier

iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
n_classes = 3

def test_simple_fit_predict():
    X = np.array([-1, -1, -1,-1, 1, 1, 1, 1]).reshape(-1,1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf = EMRVC()
    pred = clf.fit(X,y).predict(X)
    assert pred.all() == y.all()


def test_multiclass_fit_predict():
    ovr = OneVsRestClassifier(EMRVC())
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert len(ovr.estimators_) == n_classes

    clf = EMRVC()
    pred2 = clf.fit(iris.data, iris.target).predict(iris.data)
    assert np.mean(iris.target == pred) == np.mean(iris.target == pred2)

    ovr = OneVsRestClassifier(EMRVC())
    pred = ovr.fit(iris.data, iris.target).predict(iris.data)
    assert np.mean(iris.target == pred) > 0.65

