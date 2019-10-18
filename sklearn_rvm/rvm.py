"""
Relevance vector machine.
"""
# Author: Pedro Ferreira da Costa
#         Walter Hugo Lopez Pinaya
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod
from collections import deque
import math
import warnings

import numpy as np
from numpy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels

INFINITY = 1e20


class BaseRVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for relevance vector machines"""

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0,
                 tol, threshold_alpha,
                 class_weight, verbose, max_iter, random_state):

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
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

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


class RVR(BaseRVM, RegressorMixin):
    """Relevance Vector Regressor.

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

    tol : float, optional (default=1e-6)
        Tolerance for stopping criterion.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    threshold_alpha:

    max_iter : int, optional (default=5000)
        Hard limit on iterations within solver.

    verbose : bool
        Print message to stdin if True

    Attributes
    ----------
    relevance_ : array-like, shape = [n_relevance]
        Indices of relevance vectors.

    relevance_vectors_ : array-like, shape = [n_relevance, n_features]
        Relevance vectors (equivalent to X[relevance_]).

    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the relevance vector in the decision function.

    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.


    Examples
    --------

    Notes
    -----
    **References:**
    `Fast Marginal Likelihood Maximisation for Sparse Bayesian Models
    <http://www.miketipping.com/papers/met-fastsbl.pdf>`__
    """

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                 tol=1e-6, threshold_alpha=1e5,
                 max_iter=5000, verbose=False):

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha,
            max_iter=max_iter, verbose=verbose, class_weight=None, random_state=None)

    def _calculate_statistics(self, K, alpha, used_cond, y, sigma_squared):
        """TODO: Add documentation

        Parameters
        ----------
        K:
            Kernel matrix.
        alpha:
            Vector of weight precision values
        used_cond:
            Relevant basis vector indices
        y:
            Target vector.
        sigma_squared:
            Noise precision.

        Returns
        -------
        Sigma:
            Posterior covariance matrix for relevant bases
        mu:
            Posterior mean
        s:
            S-factors for all basis vectors
        q:
            Q-factors for all basis vectors
        Phi:
            Current relevant basis matrix
        """
        n_samples = y.shape[0]

        A = np.diag(alpha[used_cond])
        Phi = K[:, used_cond]

        tmp = A + (1 / sigma_squared) * Phi.T @ Phi
        if tmp.shape[0] == 1:
            Sigma = 1 / tmp
        else:
            try:
                Sigma = linalg.inv(tmp)
            except linalg.LinAlgError:
                Sigma = linalg.pinv(tmp)

        mu = (1 / sigma_squared) * Sigma @ Phi.T @ y

        # Update s and q
        Q = np.zeros(n_samples + 1)
        S = np.zeros(n_samples + 1)
        B = np.identity(n_samples) / sigma_squared
        for i in range(n_samples + 1):
            basis = K[:, i]

            # Using the Woodbury Identity, we obtain Eq. 24 and 25 from [1]
            tmp_1 = basis.T @ B
            tmp_2 = tmp_1 @ Phi @ Sigma @ Phi.T @ B
            Q[i] = tmp_1 @ y - tmp_2 @ y
            S[i] = tmp_1 @ basis - tmp_2 @ basis

        s = (alpha * S) / (alpha - S)
        q = (alpha * Q) / (alpha - S)

        return Sigma, mu, s, q, Phi

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
        # TODO: Add sample_weight?
        # TODO: Add fit_intercept (With and without bias)
        # TODO: Add fixed sigma_squared
        # TODO: Add compute_score similar to sklearn.linear_model.ARDRegression

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

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

        K = self._get_kernel(X)

        # Add bias (intercept)
        K = np.hstack((np.ones((n_samples, 1)), K))
        # TODO: Preprocess basis to have unit norm (Like tipping code)

        # 1. Initialize the sigma squared value
        # According to original code
        sigma_squared = (max(1e-6, np.std(y) * 0.1) ** 2)

        # 2. Initialize one alpha value and set all the others to infinity.
        alpha = INFINITY * np.ones(n_samples + 1)
        used_cond = np.zeros_like(alpha, dtype=bool)

        # As suggested in the paper, select bias to be the initial basis
        basis_idx = 0
        basis = K[:, basis_idx]
        basis_norm = linalg.norm(basis)
        alpha[basis_idx] = basis_norm ** 2 / ((linalg.norm(basis @ y) ** 2) / basis_norm ** 2 - sigma_squared)
        used_cond[basis_idx] = True

        # 3. Initialize Sigma and mu, and q and s for all bases
        Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha, used_cond, y, sigma_squared)

        # Create queue with indices to select candidates for update
        queue = deque(list(range(n_samples + 1)))

        # Start updating the model iteratively
        for iter in range(self.max_iter):
            if self.verbose:
                print('Iteration: {}'.format(iter))
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            old_alpha = np.copy(alpha)
            old_used_cond = np.copy(used_cond)

            reestimate_action = False

            # 5. Compute theta
            theta = q ** 2 - s

            # 6. Re-estimate alpha
            if theta[basis_idx] > 0 and alpha[basis_idx] < INFINITY:
                alpha[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                reestimate_action = True

            # 7. Add basis function to the model with updated alpha
            elif theta[basis_idx] > 0 and alpha[basis_idx] >= INFINITY:
                alpha[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                used_cond[basis_idx] = True

            # 8. Delete theta basis function from model and set alpha to infinity
            elif theta[basis_idx] <= 0 and alpha[basis_idx] < INFINITY:
                # Prevent bias to be deleted
                if basis_idx == 0:
                    continue
                alpha[basis_idx] = INFINITY
                used_cond[basis_idx] = False

            # 11. Check for convergence
            # According to the original code, step 11 is placed here.
            delta = math.log(alpha[basis_idx]) - math.log(old_alpha[basis_idx])

            if self.verbose:
                print('alpha: {}'.format(alpha))
                print('sigma_squared: {}'.format(sigma_squared))
                print('SIGMA:')
                print(Sigma)
                print('mu:')
                print(mu)
                print('theta: {}'.format(theta))
                print('delta: {}'.format(delta))
                print('Re-estimation: {}'.format(reestimate_action))

            not_used_cond = np.logical_not(old_used_cond)
            if reestimate_action and delta < self.tol and all(th <= 0 for th in theta[not_used_cond]):
                break

            # 9. Estimate noise level
            # Using updated Sigma and mu
            A = np.diag(alpha[used_cond])
            Phi = K[:, used_cond]
            tmp = A + (1 / sigma_squared) * Phi.T @ Phi
            if tmp.shape[0] == 1:
                Sigma = 1 / tmp
            else:
                try:
                    Sigma = linalg.inv(tmp)
                except linalg.LinAlgError:
                    Sigma = linalg.pinv(tmp)
            mu = (1 / sigma_squared) * Sigma @ Phi.T @ y

            # Using format from the fast algorithm paper
            y_pred = np.dot(Phi, mu)
            sigma_squared = (linalg.norm(y - y_pred) ** 2) / \
                            (n_samples - np.sum(used_cond) + np.sum(
                                np.multiply(alpha[used_cond], np.diag(Sigma))))

            # 10. Recompute/update Sigma and mu as well as s and q
            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha, used_cond, y, sigma_squared)

        # TODO: Review this part
        relevant = alpha < self.threshold_alpha
        relevant = relevant * used_cond

        self.relevance_ = np.array(list(range(len(relevant[1:]))))[relevant[1:]]
        self.relevance_vectors_ = X[self.relevance_]

        alpha_used = alpha[used_cond]
        relevant_cond = alpha_used < self.threshold_alpha

        self.mu_ = mu[relevant_cond]
        # TODO: Verify if bias is included in coef_
        self.coef_ = np.dot(self.mu_[1:], self.relevance_vectors_)
        self.dual_coef_ = self.mu_[1:]

        self.Sigma_ = Sigma[relevant_cond][:, relevant_cond]
        self.sigma_squared_ = sigma_squared

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
        check_is_fitted(self, ['relevance_vectors_', 'mu_', 'Sigma_', 'sigma_squared_'])

        X = check_array(X)

        n_samples = X.shape[0]
        K = self._get_kernel(X, self.relevance_vectors_)
        N_rv = np.shape(self.relevance_vectors_)[0]
        if np.shape(self.mu_)[0] != N_rv:
            K = np.hstack((np.ones((n_samples, 1)), K))
        y_mean = np.dot(K, self.mu_)
        if return_std is False:
            return y_mean
        else:
            err_var = self.sigma_squared_ + K @ self.Sigma_ @ K.T
            y_std = np.sqrt(np.diag(err_var))
            return y_mean, y_std


class RVC(BaseRVM, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        check_is_fitted(self)
        pass

    def predict_proba(self, X):
        check_is_fitted(self)
        pass
