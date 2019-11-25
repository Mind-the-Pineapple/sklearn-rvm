"""
============================================================================================================
Comparison of expectation maximization like relevance vector regression and fast relevance vector regression
============================================================================================================
"""
print(__doc__)

# Authors: X
# License: BSD 3 clause

import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn_rvm import RVR
from sklearn_rvm import EMRVR

# TODO:  Checar erro com esse
# np.random.seed(0)
np.random.seed(1)
rng = np.random.RandomState(0)

# Generate sample data
X = 4 * np.pi * np.random.random(100) - 2 * np.pi
y = np.sinc(X)
y += 0.25 * (0.5 - rng.rand(X.shape[0]))  # add noise

X = X[:, None]

# Fit RVR
svr = EMRVR(kernel='rbf')
stime = time.time()
svr.fit(X, y)
print("Time for EMRVR fitting: %.3f" % (time.time() - stime))

# Fit RVR
rvr = RVR(kernel='rbf')
stime = time.time()
rvr.fit(X, y)
print("Time for RVR fitting: %.3f" % (time.time() - stime))

X_plot = np.linspace(-3 * np.pi, 3 * np.pi, 10000)[:, None]
# Predict using SVR
stime = time.time()
y_svr = svr.predict(X_plot)
print("Time for EMRVR prediction: %.3f" % (time.time() - stime))

# Predict using SVR
stime = time.time()
y_svr, y_svr_std = svr.predict(X_plot, return_std=True)
print("Time for EMRVR prediction with standard-deviation: %.3f" % (time.time() - stime))

# Predict using rvm
stime = time.time()
y_rvr = rvr.predict(X_plot, return_std=False)
print("Time for RVR prediction: %.3f" % (time.time() - stime))

stime = time.time()
y_rvr, y_std = rvr.predict(X_plot, return_std=True)
print("Time for RVR prediction with standard-deviation: %.3f" % (time.time() - stime))

# Plot results
fig, axs = plt.subplots(1, 2)
lw = 2
# fig.suptitle('RVR versus EMRVR', fontsize=16)

axs[0].scatter(X, y, marker='.', c='k', label='data')
axs[0].plot(X_plot, np.sinc(X_plot), color='navy', lw=lw, label='True')

axs[0].plot(X_plot, y_svr, color='turquoise', lw=lw, label='SVR')
axs[0].fill_between(X_plot[:, 0], y_svr - y_svr_std, y_svr + y_svr_std, color='turquoise', alpha=0.2)
relevance_vectors_idx = svr.relevance_
axs[0].scatter(X[relevance_vectors_idx], y[relevance_vectors_idx], s=80, facecolors='none', edgecolors='r', label='relevance vectors')
axs[0].set_title('EMRVR')
axs[0].set_xlabel('data')
axs[0].set_ylabel('target')
axs[0].legend(loc="best")

axs[1].scatter(X, y, marker='.', c='k', label='data')
axs[1].plot(X_plot, np.sinc(X_plot), color='navy', lw=lw, label='True')

axs[1].plot(X_plot, y_rvr, color='darkorange', lw=lw, label='RVR')
axs[1].fill_between(X_plot[:, 0], y_rvr - y_std, y_rvr + y_std, color='darkorange', alpha=0.2)
relevance_vectors_idx = rvr.relevance_
axs[1].scatter(X[relevance_vectors_idx], y[relevance_vectors_idx], s=80, facecolors='none', edgecolors='r', label='relevance vectors')
axs[1].set_title('fast EM')
axs[1].set_xlabel('data')
axs[1].legend(loc="best")
plt.show()
