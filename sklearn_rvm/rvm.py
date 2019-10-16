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
    
    STEP_MIN = 1/2e8
    GRAD_MIN = 1e-6
    MAX_ITS = 100 # how to allow to change this?

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0,
                 tol=1e-6, threshold_alpha=1e5,
                 max_iter=5000, verbose=False):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha,
            max_iter=max_iter, verbose=verbose, class_weight=None, random_state=None)

    def _calculate_statistics(self, K, alpha_values, included_cond, y, mu_mp):
        """TODO: Add documentation"""
        n_samples = y.shape[0]
        error_log = []

        A = np.diag(alpha_values[included_cond])
        Phi = K[:, included_cond] # e o bias?
        M = Phi.shape[1]

        def DataErr(self, Phi, mu_mp, y):
            t_hat = np.expit(Phi @ mu_mp) # prediction of the output. Change? might be confusing
            t_hat0 = t_hat == 0
            t_hat1 = t_hat == 1

            if any(t_hat0[y>0] || any(t_hat1[y<1])):
                data_err = INFINITY
            else:
                # error is calculated through cross-entropy
                data_err = - np.sum(y*np.log(t_hat+EPSILON)) #TODO: shouldnt I divide by n_trials?
            return t_hat, data_err

        t_hat, data_err = DataErr(Phi, mu_mp, y)
        reg = A.T @ mu_mp**2 / 2
        total_err = data_err + reg
        error_log.append(total_err) # Check if cant be scalar

        for i in range(MAX_ITS):
            # Calculate the error of predictions and its gradient
            e = y - t_hat
            g = Phi.T @ e - A @ mu_mp
            # Calculate B - likelihoood dependent analogue of the noise precision 
            B = t_hat * (1-t_hat) # call it B?

            # Compute the Hessian
            tmp = Phi * (B * np.ones(1, M))
            H = tmp.T @ Phi + A
            # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
            # Must be positive definite. Check exception
            U = np.linalg.cholesky(H)

            # Check for termination based on the Gradient
            if all(abs(g))<GRAD_MIN:
                break

            # Calculate Newton Step: H^-1 * g
            delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g))
            step = 1

            while step<STEP_MIN:
                mu_new = mu + step*delta_mu
                tmp = Phi @ mu_new
                t_hat, data_err = DataErr(Phi, mu_new, y)
                reg = A.T @ mu_new**2 / 2
                total_err = data_err + reg

                # Check if error increased
                if total_err >= error_log[-1]:
                    step /= 2
                else:
                    mu_mp = mu_new
                    step = 0 # to leave the while loop

            # Compute covariance approximation
            Ui = inv(U)
            Sigma = Ui @ Ui.T

            # Update s and q
            # Check. Tippings implementation is different. Is it the same B?
            Q = np.zeros(n_samples + 1)
            S = np.zeros(n_samples + 1)
            for i in range(n_samples + 1):
                basis = K[:, i]
                # Using the Woodbury Ident ity, we obtain Eq. 24 and 25 from [1]
                tmp_1 = basis.T @ B
                tmp_2 = tmp_1 @ Phi @ Sigma @ Phi.T @ B
                Q[i] = tmp_1 @ y - tmp_2 @ y
                S[i] = tmp_1 @ basis - tmp_2 @ basis

            denominator = (alpha_values - S)
            s = (alpha_values * S) / denominator
            q = (alpha_values * Q) / denominator

            return Sigma, mu_mp, s, q, Phi
"""

        tmp = A + (1 / sigma_squared) * Phi.T @ Phi # = Phi.T @ B @ Phi + A (12)
        if tmp.shape[0] == 1:
            Sigma = 1 / tmp
        else:
            try:
                Sigma = np.linalg.inv(tmp)
            except np.linalg.LinAlgError:
                Sigma = np.linalg.pinv(tmp)

        mu_mp = (1 / sigma_squared) * Sigma @ Phi.T @ y # = Sigma @ Phi.T @ B @ t_hat (13)
        # e = t-y (14)
        # t_hat = Phi @ mu_mp + np.linalg.inv(B) @ e (14)

        # Update s and q
        Q = np.zeros(n_samples + 1)
        S = np.zeros(n_samples + 1)
        B = np.identity(n_samples) / sigma_squared # = np.diag(sigma(y(xn))* (1-sigam(y(xn))) (11)
        for i in range(n_samples + 1):
            basis = K[:, i]
            # Using the Woodbury Ident ity, we obtain Eq. 24 and 25 from [1]
            tmp_1 = basis.T @ B
            tmp_2 = tmp_1 @ Phi @ Sigma @ Phi.T @ B
            Q[i] = tmp_1 @ y - tmp_2 @ y
            S[i] = tmp_1 @ basis - tmp_2 @ basis

        denominator = (alpha_values - S)
        s = (alpha_values * S) / denominator
        q = (alpha_values * Q) / denominator

        return Sigma, mu_mp, s, q, Phi"""

    def fit(self, X, y):
        
        """TODO: Add documentation"""
        # TODO: Add fit_intercept (With and without bias)
        # TODO: Add fixed sigma_squared

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)

        n_samples = X.shape[0]
        K = pairwise_kernels(X, metric='linear', filter_params=True) # linear?
        K = np.hstack((np.ones((n_samples, 1)), K))

        # 1. Initialize the sigma squared value
        #sigma_squared = np.var(y) * 0.1

        # 2. Initialize one alpha value and set all the others to infinity.
        # Initialize mu.
        alpha_values = np.zeros(n_samples + 1) + INFINITY
        included_cond = np.zeros(n_samples + 1, dtype=bool)

        selected_basis = 0
        Phi = K[:, selected_basis] 
        t_hat = 2*y - 1 # PseudoLinear Target {-1.1}
        logout = (t_hat*0.9 + 1) / 2
        mu = np.linalg.lstsq(Phi, np.log(logout/(1-logout))) # TODO: least squares solution. np.log or math.log?
        mask_mu = mu == 0 # TODO: confirm if correct
        alpha_values[selected_basis] = 1 / (mu + mu[mask_mu])**2

        included_cond[selected_basis] = True

        # 3. Initialize Sigma, q and s for all bases
        Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)

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

            # What to do for classification?
            # 9. Estimate noise level
            # Format from the fast paper
            #y_pred = np.dot(Phi, mu)
            #sigma_squared = (np.linalg.norm(y - y_pred) ** 2) / \
                            #(n_samples - np.sum(included_cond) + np.sum(
                                #np.multiply(alpha_values[included_cond], np.diag(Sigma))))

            # 10. Recompute/update Sigma and mu as well as s and q
            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, sigma)

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

    # TODO: Check initialization of basis_vectors (4.1)
    # Selection can be random or calculate alpha and theta for all basis with more computational effort
    def predict(self, prob):
        """TODO: Add documentation"""
        return expit(prob)

    def predict_proba(self, X): # SCHUMIK line 672
        """TODO: Add documentation"""
        check_is_fitted(self, ['relevance_vectors_', 'mu_'])
        X = check_array(X)

        n_samples = X.shape[0]
        K = self._get_kernel(X, self.relevance_vectors_)
        N_rv = np.shape(self.relevance_vectors_)[0]
        if np.shape(self.mu_)[0] != N_rv:
            K = np.hstack((np.ones((n_samples, 1)), K))
        prob = K.dot(self.mu_)

        #normalize?
        #var = np.sum(np.dot(X,self.sigma)*X,1)
        #ks  = 1. / ( 1. + np.pi * var/ 8)**0.5
        #prob  = expit(y_hat * ks)
        #return prob
        return prob

     def compute_pred(self, Phi, mu): # to be replaced in the future
        return np.dot(Phi, mu)
