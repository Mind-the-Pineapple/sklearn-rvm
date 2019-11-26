"""
Relevance vector machine using expectation maximization like algorithm.

Based on
https://github.com/JamesRitchie/scikit-rvm
https://github.com/ctgk/PRML/blob/master/prml/kernel/relevance_vector_regressor.py

"""
# Author: Pedro Ferreira da Costa
#         Walter Hugo Lopez Pinaya
# License: BSD 3 clause
import warnings

from numpy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_X_y


class BaseRVM(BaseEstimator):
    """Relevance Vector Classifier.

    Parameters
    ----------
    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf' or 'sigmoid'.
         If none is given, 'rbf' will be used.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        Current default is 'auto' which uses 1 / n_features,
        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, 'auto', will change
        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
        'auto' is used as a default indicating that no explicit value of gamma
        was passed.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    threshold_alpha:

    alpha_max:

    init_alpha:

    bias_used:

    max_iter : int, optional (default=5000)
        Hard limit on iterations within solver.

    verbose : bool
        Print message to stdin if True

    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False.


    Attributes
    ----------
    relevance_ : array-like, shape = [n_relevance]
        Indices of relevance vectors.

    relevance_vectors_ : array-like, shape = [n_relevance, n_features]
        Relevance vectors (equivalent to X[relevance_]).

    alpha_:

    gamma_:

    Phi_:

    Sigma_:

    mu_:

    coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `mu` and
        `relevance_vectors_`.


    """

    def __init__(
            self,
            kernel='rbf',
            degree=3,
            gamma=None,
            coef0=0.0,
            max_iter=3000,
            tol=1e-3,
            alpha=1e-6,
            alpha_max=1e9,
            threshold_alpha=1e9,
            beta=1.e-6,
            beta_fixed=False,
            bias_used=True,
            verbose=False,
            compute_score=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.alpha_max = alpha_max
        self.threshold_alpha = threshold_alpha
        self.compute_score = compute_score
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.bias_used = bias_used
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a linear kernel')

        coef = self._get_coef()
        return coef

    def _get_coef(self):
        return np.dot(self.mu_, self.relevance_vectors_)

    def _get_kernel(self, X, Y=None):
        """Calculates kernelised features"""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self._gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'threshold_alpha': self.threshold_alpha,
            'beta': self.beta,
            'beta_fixed': self.beta_fixed,
            'bias_used': self.bias_used,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x, y):
        """Apply the selected kernel function to the data."""
        if self.kernel == 'linear':
            phi = linear_kernel(x, y)
        elif self.kernel == 'rbf':
            phi = rbf_kernel(x, y, self.gamma)
        elif self.kernel == 'poly':
            phi = polynomial_kernel(x, y, self.degree, self.gamma, self.coef0)
        elif callable(self.kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
        else:
            raise ValueError("Kernel selection is invalid.")

        if self.bias_used:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)

        return phi

    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            self.relevance_ = self.relevance_[keep_alpha[:-1]]
        else:
            self.relevance_ = self.relevance_[keep_alpha]

        self.alpha_ = self.alpha_[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma_ = self.gamma_[keep_alpha]
        self.phi = self.phi[:, keep_alpha]
        self.sigma_ = self.sigma_[np.ix_(keep_alpha, keep_alpha)]
        self.mu_ = self.mu_[keep_alpha]

    def compute_marginal_likelihood(self, n_samples, y):
        '''Calculates marginal likelihood.'''

        ED = np.sum((y - self.phi @ self.mu_) ** 2)
        U = np.linalg.cholesky(self._hessian(self, self.mu_, self.alpha_, self.phi, self.t))
        try:
            Ui = np.linalg.inv(U)
        except linalg.LinAlgError:
            Ui = np.linalg.pinv(U)
        dataLikely = (n_samples * np.log(self.beta_) - self.beta_ * ED) / 2
        logdetH = -2 * np.sum(np.log(np.diag(Ui)))
        marginal = dataLikely - 0.5 * (logdetH - np.sum(np.log(self.alpha_)) + (self.mu_ ** 2).T @ self.alpha_)
        return marginal

    def fit(self, X, y):
        """Fit the RVM to the training data."""
        X, y = check_X_y(X, y)

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if self.gamma in ('scale', 'auto_deprecated'):
            X_var = X.var()
            if self.gamma == 'scale':
                if X_var != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_var)
                else:
                    self._gamma = 1.0

            else:
                kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                                     not in ('linear', 'precomputed'))
                if kernel_uses_gamma and not np.isclose(X_var, 1.0):
                    # NOTE: when deprecation ends we need to remove explicitly
                    # setting `gamma` in examples (also in tests). See
                    # https://github.com/scikit-learn/scikit-learn/pull/10331
                    # for the examples/tests that need to be reverted.
                    warnings.warn("The default value of gamma will change "
                                  "from 'auto' to 'scale' in version 0.22 to "
                                  "account better for unscaled features. Set "
                                  "gamma explicitly to 'auto' or 'scale' to "
                                  "avoid this warning.", FutureWarning)
                self._gamma = 1.0 / X.shape[1]
        elif self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        self.scores_ = list()

        n_samples, n_features = X.shape

        self.phi = self._apply_kernel(X, X)

        n_basis_functions = self.phi.shape[1]

        self.relevance_ = X
        self.y = y

        self.alpha_ = self.alpha * np.ones(n_basis_functions)
        self.beta_ = self.beta

        self.mu_ = np.zeros(n_basis_functions)

        self.alpha_old = self.alpha_

        for i in range(self.max_iter):
            self._posterior()

            # Well-determinedness parameters (gamma)
            self.gamma_ = 1 - self.alpha_ * np.diag(self.sigma_)
            self.alpha_ = self.gamma_ / (self.mu_ ** 2)
            self.alpha_ = np.clip(self.alpha_, 0, self.alpha_max)

            if not self.beta_fixed:
                self.beta_ = (n_samples - np.sum(self.gamma_)) / (
                    np.sum((y - np.dot(self.phi, self.mu_)) ** 2))

            if self.compute_score:
                ll = self.compute_marginal_likelihood(n_samples, y)
                self.scores_.append(ll)

            self._prune()

            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma_))
                print("m: {}".format(self.mu_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
                print()

            delta = np.amax(np.absolute(self.alpha_ - self.alpha_old))

            if delta < self.tol and i > 1:
                break

            self.alpha_old = self.alpha_.copy()

        if self.bias_used:
            self.bias = self.mu_[-1]
        else:
            self.bias = None

        return self


class RVR(BaseRVM, RegressorMixin):
    """Relevance Vector Machine Regression.
    Implementation of Mike Tipping's Relevance Vector Machine for regression
    using the scikit-learn API.
    """

    def _posterior(self):
        """Compute the posterior distriubtion over weights."""
        i_s = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi.T, self.phi)
        self.sigma_ = np.linalg.inv(i_s)
        self.m_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi.T, self.y))

    def predict(self, X, eval_MSE=False):
        """Evaluate the RVR model at x."""
        phi = self._apply_kernel(X, self.relevance_)

        y = np.dot(phi, self.m_)

        if eval_MSE:
            MSE = (1 / self.beta_) + np.dot(phi, np.dot(self.sigma_, phi.T))
            return y, MSE[:, 0]
        else:
            return y


class RVC2(BaseRVM, ClassifierMixin):
    """Relevance Vector Machine Classification.
    Implementation of Mike Tipping's Relevance Vector Machine for
    classification using the scikit-learn API.
    """

    def __init__(self, n_iter_posterior=50, **kwargs):
        """Copy params to object properties, no validation."""
        self.n_iter_posterior = n_iter_posterior
        super(RVC2, self).__init__(**kwargs)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = super(RVC2, self).get_params(deep=deep)
        params['n_iter_posterior'] = self.n_iter_posterior
        return params

    def _classify(self, mu, phi):
        return expit(np.dot(phi, mu))

    def _log_posterior(self, mu, alpha, phi, t):

        y = self._classify(mu, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1 - y[t == 0]), 0))
        log_p = log_p + 0.5 * np.dot(mu.T, np.dot(np.diag(alpha), mu))

        jacobian = np.dot(np.diag(alpha), mu) - np.dot(phi.T, (t - y))

        return log_p, jacobian

    def _hessian(self, mu, alpha, phi, t):
        y = self._classify(mu, phi)
        B = np.diag(y * (1 - y))
        return np.diag(alpha) + np.dot(phi.T, np.dot(B, phi))

    def _posterior(self):
        result = minimize(
            fun=self._log_posterior,
            hess=self._hessian,
            x0=self.mu_,
            args=(self.alpha_, self.phi, self.t),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': self.n_iter_posterior
            }
        )

        self.mu_ = result.x
        self.sigma_ = np.linalg.inv(
            self._hessian(self.mu_, self.alpha_, self.phi, self.t)
        )

    def fit(self, X, y):
        """Check target values and fit model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")
        elif n_classes == 2:
            self.t = np.zeros(y.shape)
            self.t[y == self.classes_[1]] = 1
            return super(RVC2, self).fit(X, self.t)
        else:
            self.multi_ = None
            self.multi_ = OneVsOneClassifier(self)
            self.multi_.fit(X, y)
            return self

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        phi = self._apply_kernel(X, self.relevance_)
        y = self._classify(self.mu_, phi)
        return np.column_stack((1 - y, y))

    def predict(self, X):
        """Return an array of classes for each input."""
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            res = np.empty(y.shape[0], dtype=self.classes_.dtype)
            res[y[:, 1] <= 0.5] = self.classes_[0]
            res[y[:, 1] >= 0.5] = self.classes_[1]
            return res
        else:
            return self.multi_.predict(X)
