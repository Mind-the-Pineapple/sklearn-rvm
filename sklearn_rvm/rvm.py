"""
Relevance vector machine.
"""
# Author: Pedro Ferreira da Costa
#         Walter Hugo Lopez Pinaya
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels


class BaseRVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for relevance vector machines"""

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0,
                 class_weight, verbose, random_state):

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state

    @property
    def _pairwise(self):
        # Used by cross_val_score.
        return self.kernel == "precomputed"

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a '
                                 'linear kernel')

        coef = self._get_coef()

        # coef_ being a read-only property, it's better to mark the value as
        # immutable to avoid hiding potential bugs for the unsuspecting user.
        if sp.issparse(coef):
            # sparse matrix do not have global flags
            coef.data.flags.writeable = False
        else:
            # regular dense array
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)

    @property
    def n_support_(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError

        return self._n_support

    def predict(self, X):
        """Predict using the linear model
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X

# https://github.com/scikit-learn/scikit-learn/blob/0eebade2642b7f6c5105538eedada5b731998cc2/sklearn/kernel_ridge.py
class RVR(BaseRVM, RegressorMixin):
    """Relevance Vector Regressor.

    Parameters
    ----------

    Examples
    --------
    """

    def __init__(self, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, solver_type=None, verbose=False):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, verbose=verbose,
            class_weight=None, random_state=None)

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    # @property
    # def _pairwise(self):
    #     return self.kernel == "precomputed"

    def getGammaValues(self, alpha_values, Sigma):
        """Evaluates the gamma values.

        Args:
            alpha_values: N-dimensional vector with the hyperparameters of
                the marginal likelihood.
            Sigma: NxN covariance matrix of the posterior
        Returns: A N-dimensional vector with the gamma values where
            gamma_values[i] = 1 - alpha_values[i] * Sigma[i][i]
        """
        N = alpha_values.shape[0]
        gamma_values = 1 - np.multiply(alpha_values, np.diag(Sigma))
        return gamma_values

    def getAlphaValues(self, Sigma, mu, gamma_values):
        """Evaluates the alpha values.
        Args:
            Sigma: NxN covariance matrix of the posterior
            mu: mean of the posterior
            gamma_values: N-dimensional vector with gamma_values
        Returns: A N-dimensional vector with the alpha_values
        """
        N = Sigma.shape[0]
        alpha_values = np.zeros(N)
        if self.rvmType == "EM":
            cond_low = (np.diag(Sigma) + mu ** 2) < self.EPSILON_UF
            cond_high = (np.diag(Sigma) + mu ** 2) > self.INFINITY
            ncond = np.logical_and(np.logical_not(cond_low), np.logical_not(cond_high))
            alpha_values[cond_low] = self.INFINITY
            alpha_values[cond_high] = 0
            alpha_values[ncond] = 1 / (np.diag(Sigma)[ncond] + mu[ncond] ** 2)
        elif self.rvmType == "DD":
            cond_low = (mu ** 2) < self.EPSILON_UF
            cond_high = (mu ** 2) > self.INFINITY
            ncond = np.logical_and(np.logical_not(cond_low), np.logical_not(cond_high))
            alpha_values[cond_low] = self.INFINITY
            alpha_values[cond_high] = 0
            alpha_values[ncond] = gamma_values[ncond] / (mu[ncond] ** 2)
        return alpha_values

    def fit(self, X, y, sample_weight=None):
        """"""
        # From rigde kernel
        # X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
        #                  y_numeric=True)
        # if sample_weight is not None and not isinstance(sample_weight, float):
        #     sample_weight = check_array(sample_weight, ensure_2d=False)


        rnd = check_random_state(self.random_state)

        # sparse = sp.isspmatrix(X)

        X, y = check_X_y(X, y)


        self.X_fit_ = X


        # Get number of training data samples
        N = X.shape[0]
        # Initialize the sigma squared value and the B matrix
        sigma_squared = np.var(y) * 0.1
        B = np.identity(N) / sigma_squared
        # Calculate kernel matrix K and append a column with ones in the front
        K = self._get_kernel(X)
        K = np.hstack((np.ones(N).reshape((N, 1)), K))

        # 2. Initialize one alpha value and set all the others to infinity.
        alpha_values = np.zeros(N + 1) + self.INFINITY
        basis_column = K[:, 0]
        phi_norm = np.linalg.norm(basis_column)
        alpha_values[0] = (phi_norm ** 2) / (
                    (np.linalg.norm(basis_column.dot(Y_tr)) ** 2) / (phi_norm ** 2) - sigma_squared)
        included_cond = np.zeros(N + 1, dtype=bool)
        included_cond[0] = True

        # 3. Initialize Sigma and mu
        A = np.zeros(1) + alpha_values[0]
        basis_column = basis_column.reshape((N, 1))  # Reshape so that it can be transposed
        Sigma = 1 / (basis_column.T.dot(B).dot(basis_column) + A)
        mu = Sigma.dot(basis_column.T).dot(B).dot(Y_tr)

        # 3. Initialize q and s for all bases
        q = np.zeros(N + 1)
        Q = np.zeros(N + 1)
        s = np.zeros(N + 1)
        S = np.zeros(N + 1)
        Phi = basis_column
        for i in range(N + 1):
            basis = K[:, i]
            tmp_1 = basis.T.dot(B)
            tmp_2 = tmp_1.dot(Phi).dot(Sigma).dot(Phi.T).dot(B)
            Q[i] = tmp_1.dot(Y_tr) - tmp_2.dot(Y_tr)
            S[i] = tmp_1.dot(basis) - tmp_2.dot(basis)
        denom = (alpha_values - S)
        s = (alpha_values * S) / denom
        q = (alpha_values * Q) / denom

        # Create queue with indices to select candidates for update
        queue = deque([i for i in range(N + 1)])
        # Start updating the model iteratively
        for epoch in range(self.maxEpochs):
            # 4. Pick a candidate basis vector from the start of the queue and put it at the end
            basis_idx = queue.popleft()
            queue.append(basis_idx)

            # 5. Compute theta
            theta = q ** 2 - s

            next_alpha_values = np.copy(alpha_values)
            next_included_cond = np.copy(included_cond)
            if theta[basis_idx] > 0 and alpha_values[basis_idx] < self.INFINITY:
                # 6. Re-estimate alpha
                next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                pass
            elif theta[basis_idx] > 0 and alpha_values[basis_idx] >= self.INFINITY:
                # 7. Add basis function to the model with updated alpha
                next_alpha_values[basis_idx] = s[basis_idx] ** 2 / (q[basis_idx] ** 2 - s[basis_idx])
                next_included_cond[basis_idx] = True
            elif theta[basis_idx] <= 0 and alpha_values[basis_idx] < self.INFINITY:
                # 8. Delete theta basis function from model and set alpha to infinity
                next_alpha_values[basis_idx] = self.INFINITY
                next_included_cond[basis_idx] = False

            # 9. Estimate noise level
            gamma_values = 1 - np.multiply(alpha_values[included_cond], np.diag(Sigma))
            next_sigma_squared = (np.linalg.norm(Y_tr - Phi.dot(mu)) ** 2) / (N - np.sum(gamma_values))

            # 11. Check for convergence
            # Check if algorithm has converged (variation of alpha and sigma)
            not_included_cond = np.logical_not(included_cond)
            if (np.sum(np.absolute(
                    next_alpha_values[included_cond] - alpha_values[included_cond])) < self.EPSILON_CONV) and all(
                    th <= 0 for th in theta[not_included_cond]):
                break

            # 10. Recompute/update  Sigma and mu as well as s and q
            alpha_values = next_alpha_values
            sigma_squared = next_sigma_squared
            included_cond = next_included_cond
            A = np.diag(alpha_values[included_cond])
            B = np.identity(N) / sigma_squared
            Phi = K[:, included_cond]
            # Compute Sigma
            tmp = Phi.T.dot(B).dot(Phi) + A
            if (tmp.shape[0] == 1):
                Sigma = 1 / tmp
            else:
                try:
                    Sigma = np.linalg.inv(tmp)
                except linalg.LinAlgError:
                    Sigma = np.linalg.pinv(tmp)

            # Compute mu
            mu = Sigma.dot(Phi.T).dot(B).dot(Y_tr)
            # Update s and q
            for i in range(N + 1):
                basis = K[:, i]
                tmp_1 = basis.T.dot(B)
                tmp_2 = tmp_1.dot(Phi).dot(Sigma).dot(Phi.T).dot(B)
                Q[i] = tmp_1.dot(Y_tr) - tmp_2.dot(Y_tr)
                S[i] = tmp_1.dot(basis) - tmp_2.dot(basis)
            denom = (alpha_values - S)
            s = (alpha_values * S) / denom
            q = (alpha_values * Q) / denom
        ##print(epoch)
        # We store the relevance vectors and other important variables
        alpha_values = alpha_values[included_cond]
        X_tr = X_tr[included_cond[1:N + 1]]
        Y_tr = Y_tr[included_cond[1:N + 1]]

        cond_sv = alpha_values < self.TH_RV
        if alpha_values.shape[0] != X_tr.shape[0]:
            self.X_sv = X_tr[cond_sv[1:N + 1]]
            self.Y_sv = Y_tr[cond_sv[1:N + 1]]
        else:
            self.X_sv = X_tr[cond_sv]
            self.Y_sv = Y_tr[cond_sv]

        self.mu = mu[cond_sv]
        self.Sigma = Sigma[cond_sv][:, cond_sv]
        self.sigma_squared = sigma_squared

    self.bTrained = True

        # kernel = self.kernel
        # if callable(kernel):
        #     kernel = 'precomputed'

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')

        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)

        # TODO: Checar sample_weight
        # TODO: Checar precomputed kernel

        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                # var = E[X^2] - E[X]^2 if sparse
                X_var = ((X.multiply(X)).mean() - (X.mean()) ** 2
                         if sparse else X.var())
                self._gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
            elif self.gamma == 'auto':
                self._gamma = 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(self.gamma)
                )
        else:
            self._gamma = self.gamma

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            return_mean=True)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            def matvec(b):
                return X.dot(b) - b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * np.sum(b)

            X_centered = sparse.linalg.LinearOperator(shape=X.shape,
                                                      matvec=matvec,
                                                      rmatvec=rmatvec)

            if y.ndim < 2:
                out = sparse_lsqr(X_centered, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1]))
                self.coef_ = np.vstack([out[0] for out in outs])
                self._residues = np.vstack([out[3] for out in outs])
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = \
                linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class RVC(BaseRVM, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
