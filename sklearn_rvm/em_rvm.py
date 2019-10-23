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

import numpy as np
from numpy import linalg
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels


class EMRVR(RegressorMixin):
    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                 tol=1e-3, threshold_alpha=1e5, compute_score=False,
                 max_iter=5000, verbose=False, bias_used=True):

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.threshold_alpha = threshold_alpha
        self.compute_score = compute_score
        self.max_iter = max_iter
        self.verbose = verbose
        self.bias_used = bias_used

    def _get_kernel(self, X, Y=None):
        '''Calculates kernelised features'''
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self._gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True

        if self.bias_used:
            if not keep_alpha[0]:
                self.bias_used = False
            self.relevance_vectors_ = self.relevance_vectors_[keep_alpha[1:]]
        else:
            self.relevance_vectors_ = self.relevance_vectors_[keep_alpha]

        self.alpha = self.alpha[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma_ = self.gamma_[keep_alpha]
        self.Phi = self.Phi[:, keep_alpha]
        self.Sigma = self.Sigma[np.ix_(keep_alpha, keep_alpha)]
        self.mu = self.mu[keep_alpha]

    def fit(self, X, y):
        """Fit the SVM model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        Returns
        -------
        self : object
        """

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2, dtype='float64')

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

        n_samples = X.shape[0]
        self.Phi = self._get_kernel(X)
        if self.bias_used:
            self.Phi = np.hstack((np.ones((n_samples, 1)), self.Phi))

        M = self.Phi.shape[1]
        self.relevance_vectors_ = X

        # Initialize the sigma squared value and the B matrix
        sigma_squared = (max(1e-6, np.std(y) * 0.1) ** 2)
        self.beta = 1 / sigma_squared

        self.alpha = np.ones(M)

        self.alpha_old = self.alpha.copy()

        for i in range(self.max_iter):
            A = np.diag(self.alpha)
            i_s = self.beta * self.Phi.T @ self.Phi + A

            # Calculate Sigma and mu based on the initialized parameters
            try:
                self.Sigma = np.linalg.inv(i_s)
            except linalg.LinAlgError:
                self.Sigma = np.linalg.pinv(i_s)

            self.mu = self.beta * (self.Sigma @ self.Phi.T @ y)

            self.gamma_ = 1 - self.alpha * np.diag(self.Sigma)

            self.alpha = self.gamma_ / (self.mu ** 2)
            self.alpha = np.clip(self.alpha, 0, 1e10)
            self.beta = (n_samples - np.sum(self.gamma_)) / (np.sum((y - self.Phi @ self.mu) ** 2))

            self._prune()

            delta = np.amax(np.absolute(self.alpha - self.alpha_old))

            if delta < self.tol and i > 1:
                break

            self.alpha_old = self.alpha.copy()

    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        """
        # Check is fit had been called
        check_is_fitted(self, ['relevance_vectors_'])

        X = check_array(X)

        n_samples = X.shape[0]

        K = self._get_kernel(X, self.relevance_vectors_)
        if self.bias_used:
            K = np.hstack((np.ones((n_samples, 1)), K))

        y_mean = K @ self.mu
        if return_std is False:
            return y_mean
        else:
            err_var = (1 / self.beta) + K @ self.Sigma @ K.T
            y_std = np.sqrt(np.diag(err_var))
            return y_mean, y_std
