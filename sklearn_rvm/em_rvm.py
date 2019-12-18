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
from abc import abstractmethod

import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics.pairwise import pairwise_kernels
import scipy.linalg


class BaseRVM(BaseEstimator):
    """Basic class for Relevance Vector Machine."""

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, threshold_alpha,
                 beta_fixed, alpha_max, init_alpha, bias_used,
                 max_iter, compute_score, verbose):

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
        self.beta_fixed = beta_fixed
        self.alpha_max = alpha_max
        self.init_alpha = init_alpha
        self.bias_used = bias_used
        self.max_iter = max_iter
        self.compute_score = compute_score
        self.verbose = verbose

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

    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True

        if self.bias_used:
            if not keep_alpha[0]:
                self.bias_used = False
            if self.kernel != "precomputed":
                self.relevance_vectors_ = self.relevance_vectors_[
                    keep_alpha[1:]]
            self.relevance_ = self.relevance_[keep_alpha[1:]]
        else:
            if self.kernel != "precomputed":
                self.relevance_vectors_ = self.relevance_vectors_[keep_alpha]
            self.relevance_ = self.relevance_[keep_alpha]

        self.alpha_ = self.alpha_[keep_alpha]
        self._alpha_old = self._alpha_old[keep_alpha]
        self.gamma_ = self.gamma_[keep_alpha]
        self.Phi_ = self.Phi_[:, keep_alpha]
        self.Sigma_ = self.Sigma_[np.ix_(keep_alpha, keep_alpha)]
        self.mu_ = self.mu_[keep_alpha]

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    @property
    def coef_(self):
        if self.kernel != "linear":
            raise AttributeError(
                "coef_ is only available when using a linear kernel")

        coef = self._get_coef()
        return coef

    def _get_coef(self):
        return np.dot(self.mu_, self.relevance_vectors_)


class EMRVR(RegressorMixin, BaseRVM):
    """Relevance Vector Regressor.

    Parameters
    ----------
    kernel : string, optional (default="rbf")
         Specifies the kernel type to be used in the algorithm.
         It must be one of "linear", "poly", "rbf" or "sigmoid".
         If none is given, "rbf" will be used.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ("poly").
        Ignored by all other kernels.

    gamma : float, optional (default="auto")
        Kernel coefficient for "rbf", "poly" and "sigmoid".

        Current default is "auto" which uses 1 / n_features,
        if ``gamma="scale"`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, "auto", will change
        to "scale" in version 0.22. "auto_deprecated", a deprecated version of
        "auto" is used as a default indicating that no explicit value of gamma
        was passed.

    tol : float, optional (default=1e-6)
        Tolerance for stopping criterion.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in "poly" and "sigmoid".

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

    See also
    --------
    EMRVC
        Relevant Vector Machine for Classification.

    References
    ----------
    .. [1] `Tipping, Michael E (2000). "The relevance vector machine."
    <http://http://papers.nips.cc/paper/1719-the-relevance-vector-machine.pdf>`
    """

    def __init__(self, kernel="rbf", degree=3, gamma="auto_deprecated",
                 coef0=0.0, tol=1e-3, threshold_alpha=1e5,
                 beta_fixed="not_fixed", alpha_max=1e9, init_alpha=None,
                 bias_used=True, max_iter=5000, compute_score=False,
                 verbose=False):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
            threshold_alpha=threshold_alpha, beta_fixed=beta_fixed,
            alpha_max=alpha_max, init_alpha=init_alpha, bias_used=bias_used,
            max_iter=max_iter, compute_score=compute_score, verbose=verbose)

    def compute_marginal_likelihood(self, upper_inv, ed, n_samples, y):
        """Calculates marginal likelihood."""
        dataLikely = (n_samples * np.log(self.beta_) - self.beta_ * ed) / 2
        logdetH = -2 * np.sum(np.log(np.diag(upper_inv)))
        marginal = dataLikely - 0.5 * (
                logdetH - np.sum(np.log(self.alpha_)) + (
                self.mu_ ** 2).T @ self.alpha_)
        return marginal

    def fit(self, X, y):
        """Fit the RVR model according to the given training data.

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

        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2,
                         dtype="float64")

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if self.gamma in ("scale", "auto_deprecated"):
            X_var = X.var()
            if self.gamma == "scale":
                if X_var != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_var)
                else:
                    self._gamma = 1.0
            else:
                kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                                     not in ("linear", "precomputed"))
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

        n_samples = X.shape[0]
        self.Phi_ = self._get_kernel(X)

        # Scale Phi based on PRoNTO implementation
        # http://www.mlnl.cs.ucl.ac.uk/pronto/
        self._scale = np.sqrt(np.sum(self.Phi_) / n_samples ** 2)
        self.Phi_ = self.Phi_ / self._scale

        if self.bias_used:
            self.Phi_ = np.hstack((np.ones((n_samples, 1)), self.Phi_))

        M = self.Phi_.shape[1]

        if self.init_alpha == None:
            self.init_alpha = 1 / M ** 2

        self.relevance_ = np.array(range(n_samples))
        if self.kernel != "precomputed":
            self.relevance_vectors_ = X
        else:
            self.relevance_vectors_ = None

        # Initialize beta (1 / sigma squared)
        if self.beta_fixed == "not_fixed":
            sigma_squared = (max(1e-6, np.std(y) * 0.1) ** 2)
            self.beta_ = 1 / sigma_squared
        else:
            self.beta_ = self.beta_fixed

        self.alpha_ = self.init_alpha * np.ones(M)

        self._alpha_old = self.alpha_.copy()

        for i in range(self.max_iter):
            A = np.diag(self.alpha_)
            hessian = self.beta_ * self.Phi_.T @ self.Phi_ + A

            # Calculate Sigma and mu
            # Use Cholesky decomposition for efficiency
            # Ref: https://arxiv.org/abs/1111.4144
            chol_fail = False
            try:
                upper = scipy.linalg.cholesky(hessian)
            except linalg.LinAlgError:
                warnings.warn("Hessian not positive definite")
                chol_fail = True

            if chol_fail:
                try:
                    self.Sigma_ = np.linalg.inv(hessian)
                except linalg.LinAlgError:
                    self.Sigma_ = np.linalg.pinv(hessian)

                self.mu_ = self.beta_ * (self.Sigma_ @ self.Phi_.T @ y)
                sigma_diag = np.diag(self.Sigma_)

            else:
                try:
                    upper_inv = np.linalg.inv(upper)
                except linalg.LinAlgError:
                    warnings.warn("Using Pseudo-Inverse")
                    upper_inv = np.linalg.pinv(upper)

                self.Sigma_ = np.dot(upper_inv, upper_inv.conjugate().T)

                self.mu_ = (upper_inv @ (
                        upper_inv.conjugate().T @ self.Phi_.T @ y)) * self.beta_

                # Equivalent sigma_diag = np.diag(self.Sigma_)
                sigma_diag = np.sum(upper_inv ** 2, axis=1)

            # Well-determinedness parameters (gamma)
            self.gamma_ = 1 - self.alpha_ * sigma_diag

            # Alpha re-estimation
            # MacKay-style update for alpha given in original NIPS paper
            self.alpha_ = self.gamma_ / (self.mu_ ** 2)

            if self.beta_fixed == "not_fixed":
                # Prediction error
                ed = (np.sum((y - self.Phi_ @ self.mu_) ** 2))
                self.beta_ = (n_samples - np.sum(self.gamma_)) / ed

            # Compute marginal likelihood
            if not chol_fail:
                if self.compute_score:
                    ll = self.compute_marginal_likelihood(upper_inv, ed,
                                                          n_samples, y)
                    self.scores_.append(ll)

            # Passes on variable information on each iteration
            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma_))
                print("mu: {}".format(self.mu_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
                if self.compute_score:
                    print("Marginal Likelihood: {}".format(ll))
                print()

            # Prune based on large values of alpha
            self._prune()

            # Terminate if the largest alpha change is smaller than threshold
            delta = np.amax(
                np.absolute(np.log(self.alpha_) - np.log(self._alpha_old)))
            if delta < self.tol and i > 1:
                break

            self._alpha_old = self.alpha_.copy()

    def predict(self, X, return_std=False):
        """Predict using the RVR model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
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
        check_is_fitted(self, ["relevance_", "mu_", "Sigma_"])

        X = check_array(X)

        n_samples = X.shape[0]
        if self.kernel != "precomputed":
            K = self._get_kernel(X, self.relevance_vectors_)
        else:
            K = X[:, self.relevance_]
        K = K / self._scale

        if self.bias_used:
            K = np.hstack((np.ones((n_samples, 1)), K))

        y_mean = K @ self.mu_
        if return_std is False:
            return y_mean
        else:
            err_var = (1 / self.beta_) + K @ self.Sigma_ @ K.T
            y_std = np.sqrt(np.diag(err_var))
            return y_mean, y_std


class EMRVC(BaseRVM, ClassifierMixin):
    """Relevance Vector Classifier.

    Implementation of Mike Tipping"s Relevance Vector Machine for
    classification using the scikit-learn API.

    Parameters
    ----------
    kernel : string, optional (default="rbf")
         Specifies the kernel type to be used in the algorithm.
         It must be one of "linear", "poly", "rbf" or "sigmoid".
         If none is given, "rbf" will be used.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ("poly").
        Ignored by all other kernels.

    gamma : float, optional (default="auto")
        Kernel coefficient for "rbf", "poly" and "sigmoid".

        Current default is "auto" which uses 1 / n_features,
        if ``gamma="scale"`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, "auto", will change
        to "scale" in version 0.22. "auto_deprecated", a deprecated version of
        "auto" is used as a default indicating that no explicit value of gamma
        was passed.

    tol : float, optional (default=1e-6)
        Tolerance for stopping criterion.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in "poly" and "sigmoid".

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

    See also
    --------
    EMRVR
        Relevant Vector Machine for Regression.

    References
    ----------
    .. [1] `Tipping, Michael E (2000). "The relevance vector machine."
    <http://http://papers.nips.cc/paper/1719-the-relevance-vector-machine.pdf>`
    """

    def __init__(self, n_iter_posterior=50, kernel="rbf", degree=3,
                 gamma="auto_deprecated", coef0=0.0, tol=1e-3,
                 threshold_alpha=1e5, beta_fixed="not_fixed", alpha_max=1e9,
                 init_alpha=None, bias_used=True, max_iter=5000,
                 compute_score=False, verbose=False):

        self.n_iter_posterior = n_iter_posterior

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
            threshold_alpha=threshold_alpha, beta_fixed=beta_fixed,
            alpha_max=alpha_max, init_alpha=init_alpha, bias_used=bias_used,
            max_iter=max_iter, compute_score=compute_score, verbose=verbose)

    def _classify(self, mu, Phi_):
        """ Perform Sigmoid Classification."""
        return expit(np.dot(Phi_, mu))

    def _log_posterior(self, mu, alpha, Phi_, t):
        """ Calculate log posterior."""
        y = self._classify(mu, Phi_)

        log_p = -1 * (np.sum(np.log(y[t == 1]), 0) +
                      np.sum(np.log(1 - y[t == 0]), 0))
        log_p = log_p + 0.5 * np.dot(mu.T, np.dot(np.diag(alpha), mu))

        jacobian = np.dot(np.diag(alpha), mu) - np.dot(Phi_.T, (t - y))

        return log_p, jacobian

    def _compute_hessian(self, mu, alpha, Phi_):
        """ Perform the Inverse of Covariance"""
        y = self._classify(mu, Phi_)
        B = np.diag(y * (1 - y))
        return np.diag(alpha) + np.dot(Phi_.T, np.dot(B, Phi_))

    def _posterior(self):
        """ Calculates the posterior likelihood."""
        result = minimize(
            fun=self._log_posterior,
            hess=self._compute_hessian,
            x0=self.mu_,
            args=(self.alpha_, self.Phi_),
            method="Newton-CG",
            jac=True,
            options={
                "maxiter": self.n_iter_posterior
            }
        )

        self.mu_ = result.x

        hessian = self._compute_hessian(self.mu_, self.alpha_, self.Phi_)

        # Calculate Sigma
        # Use Cholesky decomposition for efficiency
        # Ref: https://arxiv.org/abs/1111.4144
        chol_fail = False
        try:
            upper = scipy.linalg.cholesky(hessian)
        except linalg.LinAlgError:
            warnings.warn("Hessian not positive definite")
            chol_fail = True

        if chol_fail:
            try:
                self.Sigma_ = np.linalg.inv(hessian)
            except linalg.LinAlgError:
                warnings.warn("Using Pseudo-Inverse")
                self.Sigma_ = np.linalg.pinv(hessian)

        else:
            try:
                upper_inv = np.linalg.inv(upper)
            except linalg.LinAlgError:
                warnings.warn("Using Pseudo-Inverse")
                upper_inv = np.linalg.pinv(upper)

            self.Sigma_ = np.dot(upper_inv, upper_inv.conjugate().T)

    def fit(self, X, y):
        """Fit the RVC model according to the given training data.

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
        X, y = check_X_y(X, y, y_numeric=True, ensure_min_samples=2,
                         dtype="float64")

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if self.gamma in ("scale", "auto_deprecated"):
            X_var = X.var()
            if self.gamma == "scale":
                if X_var != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_var)
                else:
                    self._gamma = 1.0
            else:
                kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                                     not in ("linear", "precomputed"))
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

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)

        self.scores_ = list()

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")

        elif n_classes == 2:
            self.t = np.zeros(y.shape)
            self.t[y == self.classes_[1]] = 1

            n_samples = X.shape[0]
            self.Phi_ = self._get_kernel(X)

            # Scale Phi based on PRoNTO implementation
            # http://www.mlnl.cs.ucl.ac.uk/pronto/
            self._scale = np.sqrt(np.sum(self.Phi_) / n_samples ** 2)
            self.Phi_ = self.Phi_ / self._scale

            if self.bias_used:
                self.Phi_ = np.hstack((np.ones((n_samples, 1)), self.Phi_))

            M = self.Phi_.shape[1]

            self.y = y

            if self.init_alpha == None:
                self.init_alpha = 1 / M ** 2

            self.relevance_ = np.array(range(n_samples))
            if self.kernel != "precomputed":
                self.relevance_vectors_ = X
            else:
                self.relevance_vectors_ = None

            if self.beta_fixed == "not_fixed":
                # Suggested in the paper [1].
                self.beta_ = 1e-6
            else:
                self.beta_ = self.beta_fixed

            self.mu_ = np.zeros(M)

            self.alpha_ = self.init_alpha * np.ones(M)

            self.alpha_old = self.alpha_

            for i in range(self.max_iter):
                self._posterior()

                # Well-determinedness parameters (gamma)
                self.gamma_ = 1 - self.alpha_ * np.diag(self.Sigma_)
                self.alpha_ = self.gamma_ / (self.mu_ ** 2)
                self.alpha_ = np.clip(self.alpha_, 0, self.alpha_max)

                if not self.beta_fixed:
                    self.beta_ = (n_samples - np.sum(self.gamma_)) / (
                        np.sum((y - np.dot(self.Phi_, self.mu_)) ** 2))

                if self.compute_score:
                    raise ("Score not yet implemented.")

                self._prune()

                if self.verbose:
                    print("Iteration: {}".format(i))
                    print("Alpha: {}".format(self.alpha_))
                    print("Beta: {}".format(self.beta_))
                    print("Gamma: {}".format(self.gamma_))
                    print("m: {}".format(self.mu_))
                    print("Relevance Vectors: {}".format(
                        self.relevance_.shape[0]))
                    if self.compute_score:
                        pass
                    print()

                delta = np.amax(
                    np.absolute(np.log(self.alpha_) - np.log(self._alpha_old)))

                if delta < self.tol and i > 1:
                    break

                self.alpha_old = self.alpha_.copy()

            return self

        else:
            self.multi_ = None
            self.multi_ = OneVsOneClassifier(self)
            self.multi_.fit(X, y)
            return self

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        pass

    def predict(self, X, return_std=False):
        """Predict using the RVC model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        results : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        """
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            results = np.empty(y.shape[0], dtype=self.classes_.dtype)
            results[y[:, 1] <= 0.5] = self.classes_[0]
            results[y[:, 1] >= 0.5] = self.classes_[1]
            return results
        else:
            return self.multi_.predict(X)
