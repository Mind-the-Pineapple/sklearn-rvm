import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from sklearn_rvm import RVR
from sklearn_rvm import RVC


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_template_estimator(data):
    est = RVR()

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')
    assert hasattr(est, 'relevance_vectors_')
    assert hasattr(est, 'mu_')
    assert hasattr(est, 'coef_')
    assert hasattr(est, 'Sigma_')
    assert hasattr(est, 'sigma_squared_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


def test_template_classifier(data):
    X, y = data
    clf = RVC()

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
