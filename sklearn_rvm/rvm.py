"""
Relevance vector machine.
"""
# Author: Pedro Ferreira da Costa
#         Walter Hugo Lopez Pinaya
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import expit # check if we want to impor tmore stuff

INFINITY = 1e20
EPSILON = 1e-9
STEP_MIN = 1 / 2e8
GRAD_MIN = 1e-6
MAX_ITS = 100  # how to allow to change this?
tol = 1e-6
threshold_alpha = 1e5


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
    relevance_ : array-like, shape = [n_relevance]
        Indices of relevance vectors.

    relevance_vectors_ : array-like, shape = [n_relevance, n_features]
        Relevance vectors (equivalent to X[relevance_]).

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

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha,
            max_iter=max_iter, verbose=verbose, class_weight=None, random_state=None)

    def _calculate_statistics(self, K, alpha_values, included_cond, y, sigma_squared):
        """TODO: Add documentation"""
        n_samples = y.shape[0]

        A = np.diag(alpha_values[included_cond])
        Phi = K[:, included_cond]

        tmp = A + (1 / sigma_squared) * Phi.T @ Phi
        if tmp.shape[0] == 1:
            Sigma = 1 / tmp
        else:
            try:
                Sigma = np.linalg.inv(tmp)
            except np.linalg.LinAlgError:
                Sigma = np.linalg.pinv(tmp)

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

        denominator = (alpha_values - S)
        s = (alpha_values * S) / denominator
        q = (alpha_values * Q) / denominator

        return Sigma, mu, s, q, Phi

    def fit(self, X, y):
        """TODO: Add documentation"""
        # TODO: Add fit_intercept (With and without bias)
        # TODO: Add fixed sigma_squared

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

        n_samples = X.shape[0]
        K = pairwise_kernels(X, metric='linear', filter_params=True)
        K = np.hstack((np.ones((n_samples, 1)), K))

        # 1. Initialize the sigma squared value
        sigma_squared = np.var(y) * 0.1

        # 2. Initialize one alpha value and set all the others to infinity.
        alpha_values = np.zeros(n_samples + 1) + INFINITY
        included_cond = np.zeros(n_samples + 1, dtype=bool)

        selected_basis = 0
        basis_column = K[:, selected_basis]
        phi_norm_squared = np.linalg.norm(basis_column) ** 2
        alpha_values[selected_basis] = phi_norm_squared / \
                                       ((np.linalg.norm((basis_column @ y)) ** 2) / phi_norm_squared - sigma_squared)

        included_cond[selected_basis] = True

        # 3. Initialize Sigma and mu, and q and s for all bases
        Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, sigma_squared)

        # Start updating the model iteratively
        # Create queue with indices to select candidates for update
        queue = deque(list(range(n_samples + 1)))
        for epoch in range(self.max_iter):
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            # 5. Compute theta
            theta = q ** 2 - s

            current_alpha_values = np.copy(alpha_values)
            current_included_cond = np.copy(included_cond)

            # 6. Re-estimate included alpha
            if theta[basis_idx] > 0 and current_alpha_values[basis_idx] < INFINITY:
                alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])

            # 7. Add basis function to the model with updated alpha
            elif theta[basis_idx] > 0 and current_alpha_values[basis_idx] >= INFINITY:
                alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                included_cond[basis_idx] = True

            # 8. Delete theta basis function from model and set alpha to infinity
            elif theta[basis_idx] <= 0 and current_alpha_values[basis_idx] < INFINITY:
                alpha_values[basis_idx] = INFINITY
                included_cond[basis_idx] = False

            # 9. Estimate noise level
            # Format from the fast paper
            # TODO: Fix bug here
            y_pred = np.dot(Phi, mu)
            sigma_squared = (np.linalg.norm(y - y_pred) ** 2) / \
                            (n_samples - np.sum(included_cond) + np.sum(
                                np.multiply(alpha_values[included_cond], np.diag(Sigma))))

            # 10. Recompute/update Sigma and mu as well as s and q
            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, sigma_squared)

            # 11. Check for convergence
            delta = alpha_values[current_included_cond] - current_alpha_values[current_included_cond]
            not_included_cond = np.logical_not(included_cond)
            if (np.sum(np.absolute(delta)) < self.tol) and all(th <= 0 for th in theta[not_included_cond]):
                break

        # TODO: Review this part
        alpha_values = alpha_values[included_cond]
        X = X[included_cond[1:n_samples + 1]]

        cond_rv = alpha_values < self.threshold_alpha
        if alpha_values.shape[0] != X.shape[0]:
            self.relevance_vectors_ = X[cond_rv[1:n_samples + 1]]
        else:
            self.relevance_vectors_ = X[cond_rv]

        self.mu_ = mu[cond_rv]
        self.coef_ = self.mu_
        self.Sigma_ = Sigma[cond_rv][:, cond_rv]
        self.sigma_squared_ = sigma_squared

    def predict(self, X):
        """TODO: Add documentation"""
        # Check is fit had been called
        check_is_fitted(self, ['relevance_vectors_', 'mu_'])

        X = check_array(X)

        n_samples = X.shape[0]
        K = self._get_kernel(X, self.relevance_vectors_)
        N_rv = np.shape(self.relevance_vectors_)[0]
        if np.shape(self.mu_)[0] != N_rv:
            K = np.hstack((np.ones((n_samples, 1)), K))
        y = K.dot(self.mu_)
        return y


class RVC(BaseRVM, ClassifierMixin):
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
    relevance_ : array-like, shape = [n_relevance]
        Indices of relevance vectors.

    relevance_vectors_ : array-like, shape = [n_relevance, n_features]
        Relevance vectors (equivalent to X[relevance_]).

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

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha,
            max_iter=max_iter, verbose=verbose, class_weight=None, random_state=None)

    def DataErr(self, Phi, mu, y):
        t_hat = expit(Phi @ mu) # prediction of the output. Change? might be confusing
        t_hat0 = t_hat == 0
        t_hat1 = t_hat == 1

        if (t_hat0[y>0]).any() or (t_hat1[y<1]).any():
            data_err = INFINITY
        else:
            # error is calculated through cross-entropy
            data_err = - np.sum(y[:,np.newaxis]*np.log(t_hat+EPSILON)) #TODO: shouldnt I divide by n_trials?
            y_temp = y[:,np.newaxis]
            data_err = (y_temp[np.logical_not(t_hat0)])[:,np.newaxis].T @ np.log(t_hat[np.logical_not(t_hat0)] )
            data_err += (1- (y_temp[np.logical_not(t_hat1)])[:,np.newaxis].T) @ np.log(1- t_hat[np.logical_not(t_hat1)])
            data_err = -data_err
        return t_hat, data_err

    def _calculate_statistics(self, K, alpha_values, included_cond, y, mu):

        # Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)

        n_samples = y.shape[0]
        error_log = []

        A = alpha_values[included_cond]  # np.diag(alpha_values[included_cond])
        Phi = K[:, included_cond]
        M = Phi.shape[1]
        mu, _, _, _ = np.linalg.lstsq(Phi, np.log(
            logout / (1 - logout)))  # TODO: least squares solution. np.log or math.log?

        t_hat, data_err = self.DataErr(Phi, mu, y)

        reg = A.T @ mu ** 2 / 2
        total_err = data_err + reg
        error_log.append(total_err)  # Check if cant be scalar

        for i in range(MAX_ITS):
            # Calculate the error of predictions and its gradient
            e = y[:, None] - t_hat
            g = Phi.T @ e - A[:, np.newaxis] * mu
            # Calculate B - likelihoood dependent analogue of the noise precision
            B = t_hat * (1 - t_hat)  # call it B?

            # Compute the Hessian
            tmp = Phi * (B * np.ones((1, M)))
            H = tmp.T @ Phi + A
            # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
            # Must be positive definite. Check exception
            U = np.linalg.cholesky(H)

            # Check for termination based on the Gradient
            if (abs(g) < GRAD_MIN).all():
                break

            # Calculate Newton Step: H^-1 * g
            delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g)[0])[0]
            step = 1

            while step > STEP_MIN:
                mu_new = mu + step * delta_mu
                tmp = Phi @ mu_new
                t_hat, data_err = self.DataErr(Phi, mu_new, y)
                reg = A.T @ mu_new ** 2 / 2
                total_err = np.sum(data_err + reg)

                # Check if error increased
                if total_err >= error_log[-1]:
                    step /= 2
                else:
                    mu = mu_new
                    step = 0  # to leave the while loop

        # Compute covariance approximation
        Ui = np.linalg.inv(U)
        Sigma = Ui @ Ui.T

        # Compute posterior meanbased outputs
        t_hat = expit(Phi @ mu)
        e = y[:, np.newaxis] - t_hat

        # Update s and q
        tmp_1 = K.T @ (Phi * (B * np.ones((1, M))))
        S = (B.T @ K ** 2).T - np.sum((tmp_1 @ Ui) ** 2, axis=1)[:, np.newaxis]
        Q = K.T @ e

        s = np.copy(S)
        q = np.copy(Q)

        s[included_cond] = (A[:, np.newaxis] * S[included_cond]) / (A[:, np.newaxis] - S[included_cond])
        q[included_cond] = (A[:, np.newaxis] * Q[included_cond]) / (A[:, np.newaxis] - Q[included_cond])

        for k, j in enumerate(range(np.sum(included_cond))):
            if (A[:, np.newaxis][k] - S[included_cond][j]) < EPSILON:
                s[included_cond][j] = INFINITY
                q[included_cond][j] = INFINITY

        return Sigma, mu, s, q, Phi

    def fit(self, X, y):
        
        """TODO: Add documentation"""
        # TODO: Add fit_intercept (With and without bias)

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

        n_samples = X.shape[0]
        K = pairwise_kernels(X, metric='linear', filter_params=True)  # linear?
        K = np.hstack((np.ones((n_samples, 1)), K))

        INFINITY = 1e20
        EPSILON = 1e-9

        alpha_values = np.zeros(n_samples + 1) + INFINITY
        included_cond = np.zeros(n_samples + 1, dtype=bool)

        selected_basis = 0
        Phi = K[:, selected_basis]
        t_hat = 2 * y - 1  # PseudoLinear Target {-1.1}
        logout = (t_hat * 0.9 + 1) / 2
        if len(Phi.shape) == 1:
            Phi = Phi[:, np.newaxis]
            logout = logout[:, np.newaxis]

        mu, _, _, _ = np.linalg.lstsq(Phi, np.log(
            logout / (1 - logout)))  # TODO: least squares solution. np.log or math.log?
        mask_mu = mu < EPSILON

        # alpha_values[selected_basis] = 1 / (mu + mu[mask_mu])**2 # in case there is no bias
        alpha_values[selected_basis] = EPSILON  # Set alpha to zero for free-basis
        included_cond[selected_basis] = True

        Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)

        # 3. Initialize Sigma, q and s for all bases

        # Start updating the model iteratively
        # Create queue with indices to select candidates for update
        queue = deque(list(range(n_samples + 1)))
        max_iter = 50
        for epoch in range(max_iter):
            print(epoch)
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            # 5. Compute theta
            theta = q ** 2 - s

            current_alpha_values = np.copy(alpha_values)
            current_included_cond = np.copy(included_cond)

            # 6. Re-estimate included alpha
            if theta[basis_idx] > 0 and current_alpha_values[basis_idx] < INFINITY:
                alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                print("restimate")

            # 7. Add basis function to the model with updated alpha
            elif theta[basis_idx] > 0 and current_alpha_values[basis_idx] >= INFINITY:
                alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                included_cond[basis_idx] = True
                print("include")

            # 8. Delete theta basis function from model and set alpha to infinity
            elif theta[basis_idx] <= 0 and current_alpha_values[basis_idx] < INFINITY:
                if basis_idx == 0:
                    continue
                alpha_values[basis_idx] = INFINITY
                included_cond[basis_idx] = False
                print("delete")

            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)

            # 11. Check for convergence
            delta = alpha_values[current_included_cond] - current_alpha_values[current_included_cond]
            not_included_cond = np.logical_not(included_cond)
            if (np.sum(np.absolute(delta)) < tol) and all(th <= 0 for th in theta[not_included_cond]):
                break

        # TODO: Review this part
        relevant = alpha_values < threshold_alpha
        relevant = relevant * included_cond

        # TODO: Review these relevant vector without threshold
        relevance_ = np.array(list(range(len(relevant[1:]))))[relevant[1:]]
        self.relevance_vectors_ = X[relevance_]

        self.alpha_used = alpha_values[included_cond]
        self.relevant_cond = alpha_used < threshold_alpha

        # Add scales

        self.mu_ = mu[relevant_cond]
        self.dual_coef_ = mu_[1:]
        self.coef_ = self.mu_ # ??
        self.Sigma = Sigma[relevant_cond][:, relevant_cond]
        self.relevance_vectors_ = X[relevant_cond]

        return

    # TODO: Check initialization of basis_vectors (4.1)
    # Selection can be random or calculate alpha and theta for all basis with more computational effort
    def predict(self, X):
        """TODO: Add documentation"""

        K = self.relevance_vectors_ @ X
        K = K[:,np.newaxis]
        self.logit = self.mu_ @ K
        y_pred = expit(self.logit)

        return y_pred