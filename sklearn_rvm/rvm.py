"""
Relevance vector machine.
"""
# Author: Pedro Ferreira da Costa
#         Walter Hugo Lopez Pinaya
# License: BSD 3 clause
from collections import deque

from abc import ABCMeta, abstractmethod

import numpy as np
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
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    @property
    def coef_(self):
        pass

    @property
    def relevant_vectors_(self):
        pass


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

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    Attributes
    ----------

    Examples
    --------

    Notes
    -----
    **References:**
    `Fast Marginal Likelihood Maximisation forSparse Bayesian Models
    <http://www.miketipping.com/papers/met-fastsbl.pdf>`__
    """

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                 tol=1e-6, threshold_alpha=1e5,
                 max_iter=5000, verbose=False):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha,
            max_iter=max_iter, verbose=verbose, class_weight=None, random_state=None)

    def fit(self, X, y):
        """"""
        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

        n_samples, n_features = X.shape
        # coef_ = np.zeros(n_features)
        # self.X_fit_ = X

        K = self._get_kernel(X)
        K = np.hstack((np.ones((n_samples, 1)), K))

        # 1. Initialize the sigma squared value and the B matrix
        sigma_squared = np.var(y) * 0.1
        B = np.identity(n_samples) / sigma_squared

        # 2. Initialize one alpha value and set all the others to infinity.
        alpha_values = np.zeros(n_samples + 1) + INFINITY
        included_cond = np.zeros(n_samples + 1, dtype=bool)

        basis_column = K[:, 0]
        phi_norm_squared = np.linalg.norm(basis_column) ** 2
        alpha_values[0] = phi_norm_squared / \
                          ((np.linalg.norm((basis_column @ y)) ** 2) / phi_norm_squared - sigma_squared)

        included_cond[0] = True

        # 3. Initialize Sigma and mu
        A = np.array(([alpha_values[0]]))  # A = np.diag(alpha_values[included_cond])
        basis_column = basis_column.reshape((n_samples, 1))  # basis_column[:, None]
        Sigma = 1 / (A + basis_column.T @ B @ basis_column)
        mu = Sigma @ basis_column.T @ B @ y

        # 3. Initialize q and s for all bases
        Q = np.zeros(n_samples + 1)
        S = np.zeros(n_samples + 1)
        Phi = basis_column
        for i in range(n_samples + 1):
            basis = K[:, i]
            # Using the Woodbury Identity, we obtain Eq. 24 and 25 from [1]
            tmp_1 = basis.T @ B
            tmp_2 = tmp_1 @ Phi @ Sigma @ Phi.T @ B
            Q[i] = tmp_1 @ y - tmp_2 @ y
            S[i] = tmp_1 @ basis - tmp_2 @ basis

        denominator = (alpha_values - S)
        s = (alpha_values * S) / denominator
        q = (alpha_values * Q) / denominator

        # Create queue with indices to select candidates for update
        queue = deque([i for i in range(n_samples + 1)])

        # Start updating the model iteratively
        for epoch in range(self.max_iter):
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            # 5. Compute theta
            theta = q ** 2 - s

            next_alpha_values = np.copy(alpha_values)
            next_included_cond = np.copy(included_cond)

            if theta[basis_idx] > 0 and alpha_values[basis_idx] < INFINITY:
                # 6. Re-estimate alpha
                next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])

            elif theta[basis_idx] > 0 and alpha_values[basis_idx] >= INFINITY:
                # 7. Add basis function to the model with updated alpha
                next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                next_included_cond[basis_idx] = True

            elif theta[basis_idx] <= 0 and alpha_values[basis_idx] < INFINITY:
                # 8. Delete theta basis function from model and set alpha to infinity
                next_alpha_values[basis_idx] = INFINITY
                next_included_cond[basis_idx] = False

            # 9. Estimate noise level
            # TODO: verify
            gamma_values = 1 - np.multiply(alpha_values[included_cond], np.diag(Sigma))
            next_sigma_squared = (np.linalg.norm(y - np.dot(Phi, mu)) ** 2) / (n_samples - np.sum(gamma_values))

            # 11. Check for convergence
            delta = next_alpha_values[included_cond] - alpha_values[included_cond]
            not_included_cond = np.logical_not(included_cond)

            # TODO: verify change order
            # 10. Recompute/update  Sigma and mu as well as s and q
            alpha_values = next_alpha_values
            sigma_squared = next_sigma_squared
            included_cond = next_included_cond

            A = np.diag(alpha_values[included_cond])
            B = np.identity(n_samples) / sigma_squared
            Phi = K[:, included_cond]

            # Compute Sigma
            tmp = Phi.T @ B @ Phi + A
            if tmp.shape[0] == 1:
                Sigma = 1 / tmp
            else:
                try:
                    Sigma = np.linalg.inv(tmp)
                except np.linalg.LinAlgError:
                    Sigma = np.linalg.pinv(tmp)

            # Compute mu
            mu = Sigma @ Phi.T @ B @ y

            # TODO: Deal with repeated code
            # Update s and q
            for i in range(n_samples + 1):
                basis = K[:, i]
                # Using the Woodbury Identity, we obtain Eq. 24 and 25 from [1]
                tmp_1 = basis.T @ B
                tmp_2 = tmp_1 @ Phi @ Sigma @ Phi.T @ B
                Q[i] = tmp_1 @ y - tmp_2 @ y
                S[i] = tmp_1 @ basis - tmp_2 @ basis

            denominator = (alpha_values - S)
            s = (alpha_values * S) / denominator
            q = (alpha_values * Q) / denominator

            # 11. Check for convergence
            # Check if algorithm has converged (variation of alpha and sigma)
            if (np.sum(np.absolute(delta)) < self.tol) and all(th <= 0 for th in theta[not_included_cond]):
                break

        # We store the relevance vectors and other important variables
        alpha_values = alpha_values[included_cond]
        X = X[included_cond[1:n_samples + 1]]
        y = y[included_cond[1:n_samples + 1]]

        cond_sv = alpha_values < self.threshold_alpha
        if alpha_values.shape[0] != X.shape[0]:
            self.X_sv_ = X[cond_sv[1:n_samples + 1]]
            self.Y_sv_ = y[cond_sv[1:n_samples + 1]]
        else:
            self.X_sv_ = X[cond_sv]
            self.Y_sv_ = y[cond_sv]

        self.mu_ = mu[cond_sv]
        self.Sigma = Sigma[cond_sv][:, cond_sv]
        self.sigma_squared = sigma_squared

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_sv_', 'mu_'])

        X = check_array(X)

        n_samples, n_features = X.shape
        K = self._get_kernel(X, self.X_sv_)
        N_sv = np.shape(self.X_sv_)[0]
        if np.shape(self.mu_)[0] != N_sv:
            K = np.hstack((np.ones((n_samples, 1)), K))
        y = K.dot(self.mu_)
        return y


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
