from sklearn_rvm.rvm import RVC
from sklearn.datasets import load_iris
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.special import expit # check if we want to impor tmore stuff

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


class EMRVR(RegressorMixin, BaseRVM):
    """Relevance Vector Regressor.

    Implementation of the relevance vector regressor using the algorithm
    based on expectation maximization.

    Parameters
    ----------
    kernel : string, optional (default="rbf")
        Specifies the kernel type to be used in the algorithm.
        It must be one of "linear", "poly", "rbf", "sigmoid" or "precomputed".
        If none is given, "rbf" will be used.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ("poly"). Ignored by all other
        kernels.

    gamma : {"auto", "scale"} or float, optional (default="auto")
        Kernel coefficient for "rbf", "poly" and "sigmoid".

        Current default is "auto" which uses 1 / n_features,
        if ``gamma="scale"`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function. It is only significant in "poly"
        and "sigmoid".

    tol : float, optional (default=1e-6)
        Tolerance for stopping criterion.

    threshold_alpha : float, optional (default=1e5)
        Threshold for alpha selection criterion.

    beta_fixed : {"not_fixed"} or float, optional (default="not_fixed")
        Fixed value for beta. If "not_fixed" selected, the beta is updated at
        each iteration.

    alpha_max : int, optional (default=1e9)
        Basis functions associated with alpha value beyond this limit will be
        purged. Must be a positive and big number.

    init_alpha : array-like of shape (n_sample) or None, optional (default=None)
        Initial value for alpha. If None is selected, the initial value of
        alpha is defined by init_alpha = 1 / M ** 2.

    bias_used : boolean, optional (default=False)
        Specifies if a constant (a.k.a. bias) should be added to the decision
        function.

    max_iter : int, optional (default=5000)
        Hard limit on iterations within solver.

    compute_score : boolean, optional (default=False)
        Specifies if the objective function is computed at each step of the model.

    verbose : boolean, optional (default=False)
        Enable verbose output.

    Attributes
    ----------
    relevance_ : array-like, shape (n_relevance)
        Indices of relevance vectors.

    relevance_vectors_ : array-like, shape (n_relevance, n_features)
        Relevance vectors (equivalent to X[relevance_]).

    alpha_ : array-like, shape (n_samples)
        Estimated alpha values.

    gamma_ : array-like, shape (n_samples)
        Estimated gamma values.

    Phi_ : array-like, shape (n_samples, n_features)
        Estimated phi values.

    Sigma_ : array-like, shape (n_samples, n_features)
        Estimated covariance matrix of the weights.

    mu_ : array-like, shape (n_relevance, n_features)
        Coefficients of the regression model (mean of posterior distribution)

    coef_ : array, shape (n_class * (n_class-1) / 2, n_features)
        Coefficients of the regression model (mean of posterior distribution).
        Weights assigned to the features. This is only available in the case
        of a linear kernel. `coef_` is a readonly property derived from `mu`
        and `relevance_vectors_`.

    See Also
    --------
    EMRVC
        Relevant Vector Machine for Classification.

    Notes
    -----
    **References:**
    `The relevance vector machine.
    <http://www.miketipping.com/sparsebayes.htm>`__
    """

    def __init__(self, kernel="rbf", degree=3, gamma="auto_deprecated",
                 coef0=0.0, tol=1e-3, threshold_alpha=1e5,
                 beta_fixed="not_fixed", alpha_max=1e9, init_alpha=None,
                 bias_used=True, max_iter=5000, compute_score=False,
                 epsilon=1e-08, verbose=False):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
            threshold_alpha=threshold_alpha, beta_fixed=beta_fixed,
            alpha_max=alpha_max, init_alpha=init_alpha, bias_used=bias_used,
            max_iter=max_iter, compute_score=compute_score, epsilon=epsilon,
            verbose=verbose)

    INFINITY = 1e20
    EPSILON = 1e-9
    def DataErr(self, Phi, mu_mp, y):
        t_hat = expit(Phi @ mu_mp)  # prediction of the output. Change? might be confusing
        t_hat0 = t_hat == 0
        t_hat1 = t_hat == 1

        if any(t_hat0[y > 0]) or any(t_hat1[y < 1]):
            data_err = INFINITY
        else:
            # error is calculated through cross-entropy
            data_err = - np.sum(y * np.log(t_hat + EPSILON))  # TODO: shouldnt I divide by n_trials?
        return t_hat, data_err

    def _calculate_statistics(self, X, alpha_values, included_cond, y, sigma):

        # This must be added to the parameters for the function
        STEP_MIN = 1/2e8
        GRAD_MIN = 1e-6
        MAX_ITS = 100

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2)


        n_samples = X.shape[0]
        K = pairwise_kernels(X, metric='linear', filter_params=True) # linear?
        K = np.hstack((np.ones((n_samples, 1)), K))

        alpha_values = np.zeros(n_samples + 1) + INFINITY
        included_cond = np.zeros(n_samples + 1, dtype=bool)

        selected_basis = 0
        Phi = K[:, selected_basis]
        t_hat = 2*y - 1 # PseudoLinear Target {-1.1}
        logout = (t_hat*0.9 + 1) / 2
        if len(Phi.shape)==1:
            Phi = Phi[:,np.newaxis]
            logout = logout[:,np.newaxis]

        mu, _, _, _ = np.linalg.lstsq(Phi, np.log(logout/(1-logout))) # TODO: least squares solution. np.log or math.log?
        mask_mu = mu < EPSILON # TODO: confirm if correct


        alpha_values[selected_basis] = 1 / (mu + mu[mask_mu])**2

        included_cond[selected_basis] = True

        #Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)
        ####
        mu_mp = mu

        n_samples = y.shape[0]
        error_log = []

        A = np.diag(alpha_values[included_cond])
        Phi = K[:, included_cond] # e o bias?
        M = Phi.shape[1]

        t_hat, data_err = self.DataErr(Phi, mu_mp, y)
        reg = A.T @ mu_mp**2 / 2
        total_err = data_err + reg
        error_log.append(total_err) # Check if cant be scalar

    def fit(self, X, y):

        for i in range(MAX_ITS):
            # Calculate the error of predictions and its gradient
            e = y[:,None] - t_hat
            g = Phi.T @ e - A * mu_mp
            # Calculate B - likelihoood dependent analogue of the noise precision
            B = t_hat * (1-t_hat) # call it B?

            # Compute the Hessian
            tmp = Phi * (B * np.ones((1, M)))
            H = tmp.T @ Phi + A
            # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
            # Must be positive definite. Check exception
            U = np.linalg.cholesky(H)

            # Check for termination based on the Gradient
            if all(abs(g))<GRAD_MIN:
                break

            # Calculate Newton Step: H^-1 * g
            delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g)[0])[0]
            step = 1

            while step>STEP_MIN:
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
            Ui = np.linalg.inv(U)
            Sigma = Ui @ Ui.T

            # Update s and q
            tmp_1 = K.T @ (Phi * (B * np.ones((1,M))))
            S = (B.T @ K**2).T - np.sum((tmp_1 @ Ui)**2, axis=1)[:,np.newaxis]
            Q = K.T @ e

            s = S
            q = Q

            s[included_cond] = (A * S[included_cond]) / (A - S[included_cond])
            q[included_cond] = (A * Q[included_cond]) / (A - Q[included_cond])

        # 3. Initialize Sigma, q and s for all bases

        # Start updating the model iteratively
        # Create queue with indices to select candidates for update
        queue = deque(list(range(n_samples + 1)))
        max_iter = 100
        for epoch in range(max_iter):
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


            Sigma, mu, s, q, Phi = self._calculate_statistics(K, alpha_values, included_cond, y, mu)
            mu_mp = mu

            n_samples = y.shape[0]
            error_log = []

            A = np.diag(alpha_values[included_cond])
            Phi = K[:, included_cond] # e o bias?
            M = Phi.shape[1]

            def DataErr( Phi, mu_mp, y):
                t_hat = expit(Phi @ mu_mp) # prediction of the output. Change? might be confusing
                t_hat0 = t_hat == 0
                t_hat1 = t_hat == 1

                if any(t_hat0[y>0]) or any(t_hat1[y<1]):
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
                e = y[:,None] - t_hat
                g = Phi.T @ e - A * mu_mp
                # Calculate B - likelihoood dependent analogue of the noise precision
                B = t_hat * (1-t_hat) # call it B?

                # Compute the Hessian
                tmp = Phi * (B * np.ones((1, M)))
                H = tmp.T @ Phi + A
                # Invert Hessian via Cholesky - lower triangular Cholesky factor of H.
                # Must be positive definite. Check exception
                U = np.linalg.cholesky(H)

                # Check for termination based on the Gradient
                if all(abs(g))<GRAD_MIN:
                    break

                # Calculate Newton Step: H^-1 * g
                delta_mu = np.linalg.lstsq(U, np.linalg.lstsq(U.T, g)[0])[0]
                step = 1

                while step>STEP_MIN:
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
                Ui = np.linalg.inv(U)
                Sigma = Ui @ Ui.T

                # Update s and q
                tmp_1 = K.T @ (Phi * (B * np.ones((1,M))))
                S = (B.T @ K**2).T - np.sum((tmp_1 @ Ui)**2, axis=1)[:, np.newaxis]
                Q = K.T @ e

                s = S
                q = Q

                s[included_cond] = (A * S[included_cond]) / (A - S[included_cond])
                q[included_cond] = (A * Q[included_cond]) / (A - Q[included_cond])













