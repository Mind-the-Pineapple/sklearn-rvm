"""
=====================================================================
Comparison of relevance vector regression and ARDRegression
=====================================================================
"""
print(__doc__)

# Authors: X
# License: BSD 3 clause

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn_rvm import EMRVR

np.random.seed(1)
dimension = 1
noise_to_signal = 0.2
N = 100
basis_width = 0.05  # NB: data is in [0,1]

# Define probability of a basis function NOT being used by the generative
# model. i.e. if pSparse=0.90, only 10% of basis functions (on average) will
# be used to synthesise the data.
p_sparse = 0.90
iterations = 500

# Heuristically adjust basis width to account for distance scaling with dimension.
basis_width = basis_width ** (1 / dimension)

# --- SYNTHETIC DATA GENERATION ---
# First define the input data over a regular grid
X = (np.array(range(0, N)) / N)[:, None]

# Now define the basis
# Locate basis functions at data points

C = X


def dist_squared(X, Y):
    nx = X.shape[0]
    ny = Y.shape[0]
    d2 = (np.sum(np.power(X, 2), axis=1)[:, None] * np.ones((1, ny))) + \
         (np.ones((nx, 1)) * (np.sum(np.power(Y, 2)[:, None], axis=1)).T) - 2 * X * Y.T
    return d2


# Compute ("Gaussian") basis (design) matrix
basis = np.exp(-dist_squared(X, C) / (basis_width ** 2))

# Randomise some weights, then make each weight sparse with probability pSparse
M = basis.shape[1]
w = np.random.randn(M) * 100 / (M * (1 - p_sparse))
sparse = np.random.rand(M) < p_sparse
w[sparse] = 0
# Now we have the basis and weights, compute linear model
z = basis @ w
# Finally generate the data according to the likelihood model

# Generate our data by adding some noise on to the generative function
noise = np.std(z) * noise_to_signal
outputs = z + noise * np.random.randn(N)

rvr = RVR(kernel='precomputed', compute_score=True)
rvr.fit(basis, outputs)

w_infer = np.zeros((M, 1))
w_infer[rvr.relevance_, :] = rvr.dual_coef_[:, None]

y_l = basis @ w_infer

# Plot the data
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(X, outputs, marker='.')
axs[0, 0].title.set_text('Generated data ({:} points)'.format(N))

# Likelihood trace (and Gaussian noise info)
lsteps = len(rvr.scores_)
axs[0, 1].title.set_text('Log marginal likelihood trace')
axs[0, 1].plot(range(lsteps), rvr.scores_)
axs[0, 1].text(0.5, 0.5,
               'Actual noise:   {:.5f}\nInferred noise: {:.5f}'.format(noise, np.sqrt(rvr.sigma_squared_)),
               horizontalalignment='center',
               verticalalignment='center',
               transform=axs[0, 1].transAxes)

# Compare the generative and predictive linear models
axs[0, 2].plot(X, z, color='k', linewidth=4, label='Actual')
axs[0, 2].plot(X, y_l, color='r', linewidth=3.0, label='Model')
axs[0, 2].title.set_text('Generative function and linear model')
axs[0, 2].legend()

# Compare the data and the predictive model (post link-function)
axs[1, 0].scatter(X, outputs, marker='.')
axs[1, 0].plot(X, y_l, color='r')
axs[1, 0].scatter(X[rvr.relevance_], outputs[rvr.relevance_], s=80, facecolors='none', edgecolors='r', )
axs[1, 0].title.set_text('Data and predictor')

# Show the inferred weights
axs[1, 1].stem(w_infer, markerfmt='.')
axs[1, 1].title.set_text('Inferred weights ({:})'.format(len(rvr.relevance_)))

# Show the "well-determinedness" factors
axs[1, 2].bar(range(len(rvr.gamma_)), rvr.gamma_)
axs[1, 2].title.set_text('Well-determinedness (gamma)')
plt.show()
