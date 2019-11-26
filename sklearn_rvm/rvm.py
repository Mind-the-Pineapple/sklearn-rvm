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
from numpy.linalg import LinAlgError
from scipy.linalg import pinvh
from numpy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import expit 
from sklearn.multiclass import OneVsOneClassifier

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
                 tol, threshold_alpha, compute_score,
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
        self.compute_score = compute_score
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

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
        # TODO: Verify if bias is included in coef_

        return np.dot(self.mu_[1:], self.relevance_vectors_)

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

    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False.

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
                 tol=1e-6, threshold_alpha=1e5, compute_score=False,
                 max_iter=5000, verbose=False):

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, threshold_alpha=threshold_alpha, compute_score=compute_score,
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

        # TODO: Fix aligment
        # check_alignment = False

        self.scores_ = list()

        n_samples = X.shape[0]

        K = self._get_kernel(X)

        # Scale basis vectors to unit norm. This eases some calculations and will improve numerical robustness later.
        K, scales = self._preprocess_basis(K)

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

            if self.compute_score:
                # compute the log marginal likelihood
                score = self._log_marginal_likelihood(n_samples, sigma_squared, Phi, mu, alpha, y, used_cond)
                self.scores_.append(score)

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
                # if check_alignment == True:
                #     p = basis.T @ Phi
                #     p_mask = p > (1 - 1e-3)
                #     if sum(p_mask) == 0:
                #         used_cond[basis_idx] = True
                # else:
                #     used_cond[basis_idx] = True
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
                if self.verbose:
                    print("Converged after %s iterations" % iter)
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
            gamma = 1 - np.multiply(alpha[used_cond], np.diag(Sigma))
            sigma_squared = (linalg.norm(y - y_pred) ** 2) / \
                            (n_samples - np.sum(used_cond) + np.sum(
                                np.multiply(alpha[used_cond], np.diag(Sigma))))

            # 10. Recompute/update Sigma and mu as well as s and q
            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha, used_cond, y, sigma_squared)

        if self.compute_score:
            # compute the log marginal likelihood
            score = self._log_marginal_likelihood(n_samples, sigma_squared, Phi, mu, alpha, y, used_cond)
            self.scores_.append(score)

        # TODO: Review this part
        relevant = alpha < self.threshold_alpha
        relevant = relevant * used_cond

        # TODO: Review these relevant vector without threshold
        self.relevance_ = np.array(list(range(len(relevant[1:]))))[relevant[1:]]
        self.relevance_vectors_ = X[self.relevance_]

        alpha_used = alpha[used_cond]
        relevant_cond = alpha_used < self.threshold_alpha

        # Add scales
        self.scales = scales[relevant[1:]]
        self.gamma_ = gamma[1:]
        self.mu_ = mu[relevant_cond]
        self.mu_[1:] = self.mu_[1:] / self.scales
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

    def _preprocess_basis(self, K):
        N, M = K.shape

        # Compute "lengths" of basis vectors (columns of BASIS)
        scales = np.sqrt(np.sum(K ** 2, axis=0))

        # Work-around divide-by-zero inconvenience
        scales[scales == 0] = 1

        # Normalise each basis vector to "unit length"
        for m in range(M):
            K[:, m] = K[:, m] / scales[m]
        return K, scales

    def _log_marginal_likelihood(self, n_samples, sigma_squared, Phi, mu, alpha, y, used_cond):
        """Log marginal likelihood."""
        beta = (1 / sigma_squared)
        U = linalg.cholesky(Phi.T @ Phi * beta + np.diag(alpha[used_cond]))
        y_pred = Phi @ mu
        e = (y - y_pred)
        ED = e.T @ e
        dataLikely = (n_samples * np.log(beta) - beta * ED) / 2
        logdetHOver2 = np.sum(np.log(np.diag(U)))
        logML = dataLikely - (mu ** 2).T @ alpha[used_cond] / 2 + np.sum(np.log(alpha[used_cond])) / 2 - logdetHOver2
        return logML


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

        if (t_hat0[y > 0]).any() or (t_hat1[y < 1]).any():
            data_err = INFINITY
        else:
            # error is calculated through cross-entropy
            data_err = - np.sum(y[:,np.newaxis]*np.log(t_hat+EPSILON)) #TODO: shouldnt I divide by n_trials?
            y_temp = y[:,np.newaxis]
            # TODO: Check if the dot product below is element wise
            data_err = (y_temp[np.logical_not(t_hat0)])[:,np.newaxis].T @ np.log(t_hat[np.logical_not(t_hat0)] )
            data_err += (1 - (y_temp[np.logical_not(t_hat1)])[:,np.newaxis].T) @ np.log(1- t_hat[np.logical_not(t_hat1)])
            data_err = -data_err
        return t_hat, data_err

    def _calculate_statistics(self, K, alpha_values, included_cond, y, logout):

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
            try:
                U = np.linalg.cholesky(H)
            except LinAlgError:
                U = pinvh(H)
            print(g)
            print('+++++++++++++++++')
            # Check for termination based on the Gradient
            if (abs(g) < GRAD_MIN).all():
                break
            #print(np.linalg.cholesky(H))
            #print(pinvh(H))
            #print('+++++++++++++++++++++++++++++++++++++')
            # Calculate Newton Step: H^-1 * g
            delta_mu = np.linalg.solve(U, np.linalg.solve(U.T, g))
            step = 1.0

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
        # TODO: Is it Ui.T @ Ui
        Sigma = Ui @ Ui.T

        # Compute posterior meanbased outputs
        t_hat = expit(Phi @ mu)
        e = y[:, np.newaxis] - t_hat

        # Update s and q
        tmp_1 = K.T @ (Phi * (B * np.ones((1, M))))
        # TODO: Ui.T em baixo?
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

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("2 classes at least are needed to train the model.")

        elif n_classes > 2:
            self.multi_ = None
            self.multi_ = OneVsOneClassifier(self)
            self.multi_.fit(X, y)
            return self

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
        a = np.divide(logout, 1-logout)
        b = np.log(np.divide(logout, 1-logout))
        c = np.linalg.lstsq(Phi, np.log(np.divide(logout, 1-logout)))
        mu, _, _, _ = np.linalg.lstsq(Phi, np.log(np.divide(logout, 1-logout)))# TODO: least squares solution. np.log or math.log?
        mask_mu = mu < EPSILON

        # alpha_values[selected_basis] = 1 / (mu + mu[mask_mu])**2 # in case there is no bias
        alpha_values[selected_basis] = EPSILON  # Set alpha to zero for free-basis
        included_cond[selected_basis] = True

        Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, logout)

        # 3. Initialize Sigma, q and s for all bases

        # Start updating the model iteratively
        # Create queue with indices to select candidates for update
        queue = deque(list(range(n_samples + 1)))
        max_iter = 1000
        for epoch in range(max_iter):
            print(epoch)
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            # 5. Compute theta
            theta = q ** 2 - s

            current_alpha_values = np.copy(alpha_values)
            current_included_cond = np.copy(included_cond)


            ### DEBUGGING
            #print(q)
            #print(s)
            #xprint(basis_idx)
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

            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, logout)

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

        alpha_used = alpha_values[included_cond]
        self.relevant_cond = alpha_used < threshold_alpha

        # Add scales

        self.mu_ = mu[self.relevant_cond]
        self.dual_coef_ = self.mu_[1:]
        self.coef_ = self.mu_ # ??
        self.Sigma = Sigma[self.relevant_cond][:, self.relevant_cond]
        self.relevance_vectors_ = X[self.relevant_cond]

        return

    # TODO: Check initialization of basis_vectors (4.1)
    # Selection can be random or calculate alpha and theta for all basis with more computational effort
    def predict(self, X):
        """TODO: Add documentation"""
        n_samples = X.shape[0]
        K = self.relevant_cond @ X
        K = K[:,np.newaxis]
        K = np.hstack((np.ones((n_samples, 1)), K))
        self.logit = K @ self.mu_
        y_pred = expit(self.logit)

        return y_pred