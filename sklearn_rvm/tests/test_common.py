import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_rvm import EMRVR

@pytest.mark.parametrize(
    "Estimator", [EMRVR]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
