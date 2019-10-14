"""
=====================================================================
Comparison of relevance vector regression and support vector machines
=====================================================================
Both kernel ridge regression (KRR) and Gaussian process regression (GPR) learn
a target function by employing internally the "kernel trick". KRR learns a
linear function in the space induced by the respective kernel which corresponds
to a non-linear function in the original space. The linear function in the
kernel space is chosen based on the mean-squared error loss with
ridge regularization. GPR uses the kernel to define the covariance of
a prior distribution over the target functions and uses the observed training
data to define a likelihood function. Based on Bayes theorem, a (Gaussian)
posterior distribution over target functions is defined, whose mean is used
for prediction.
A major difference is that GPR can choose the kernel's hyperparameters based
on gradient-ascent on the marginal likelihood function while KRR needs to
perform a grid search on a cross-validated loss function (mean-squared error
loss). A further difference is that GPR learns a generative, probabilistic
model of the target function and can thus provide meaningful confidence
intervals and posterior samples along with the predictions while KRR only
provides predictions.
This example illustrates both methods on an artificial dataset, which
consists of a sinusoidal target function and strong noise. The figure compares
the learned model of KRR and GPR based on a ExpSineSquared kernel, which is
suited for learning periodic functions. The kernel's hyperparameters control
the smoothness (l) and periodicity of the kernel (p). Moreover, the noise level
of the data is learned explicitly by GPR by an additional WhiteKernel component
in the kernel and by the regularization parameter alpha of KRR.
The figure shows that both methods learn reasonable models of the target
function. GPR correctly identifies the periodicity of the function to be
roughly 2*pi (6.28), while KRR chooses the doubled periodicity 4*pi. Besides
that, GPR provides reasonable confidence bounds on the prediction which are not
available for KRR. A major difference between the two methods is the time
required for fitting and predicting: while fitting KRR is fast in principle,
the grid-search for hyperparameter optimization scales exponentially with the
number of hyperparameters ("curse of dimensionality"). The gradient-based
optimization of the parameters in GPR does not suffer from this exponential
scaling and is thus considerable faster on this example with 3-dimensional
hyperparameter space. The time for predicting is similar; however, generating
the variance of the predictive distribution of GPR takes considerable longer
than just predicting the mean.
"""
print(__doc__)

# Authors: X
# License: BSD 3 clause

import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn_rvm import RVR

rng = np.random.RandomState(0)

# Generate sample data
X = 2 * np.pi * np.random.random(25) - np.pi
y = np.sinc(X)
y += 0.25 * (0.5 - rng.rand(X.shape[0]))  # add noise

X = X[:, None]

# Fit RVR
svr = SVR(kernel='rbf', gamma=1)
stime = time.time()
svr.fit(X, y)
print("Time for SVR fitting: %.3f" % (time.time() - stime))

# Fit RVR
rvr = RVR(kernel='rbf', gamma=1)
stime = time.time()
rvr.fit(X, y)
print("Time for RVR fitting: %.3f" % (time.time() - stime))

X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 10000)[:, None]
# Predict using SVR
stime = time.time()
y_svr = svr.predict(X_plot)
print("Time for SVR prediction: %.3f" % (time.time() - stime))

# Predict using rvm
stime = time.time()
y_rvr = rvr.predict(X_plot, return_std=False)
print("Time for RVR prediction: %.3f" % (time.time() - stime))

stime = time.time()
y_rvr, y_std = rvr.predict(X_plot, return_std=True)
print("Time for RVR prediction with standard-deviation: %.3f" % (time.time() - stime))

# Plot results
fig = plt.figure(figsize=(10, 5))
lw = 2
fig.suptitle('RVR versus SVR', fontsize=16)

plt.subplot(121)
plt.scatter(X, y, marker='.', c='k', label='data')
plt.plot(X_plot, np.sinc(X_plot), color='navy', lw=lw, label='True')

plt.plot(X_plot, y_svr, color='turquoise', lw=lw, label='SVR')
support_vectors_idx = svr.support_
plt.scatter(X[support_vectors_idx], y[support_vectors_idx], s=80, facecolors='none', edgecolors='r', label='support vectors')
plt.ylabel('target')
plt.xlabel('data')
plt.legend(loc="best", scatterpoints=1, prop={'size': 8})

plt.subplot(122)
plt.scatter(X, y, marker='.', c='k', label='data')
plt.plot(X_plot, np.sinc(X_plot), color='navy', lw=lw, label='True')

plt.plot(X_plot, y_rvr, color='darkorange', lw=lw, label='RVR')
plt.fill_between(X_plot[:, 0], y_rvr - y_std, y_rvr + y_std, color='darkorange', alpha=0.2)
relevance_vectors_idx = rvr.relevance_
plt.scatter(X[relevance_vectors_idx], y[relevance_vectors_idx], s=80, facecolors='none', edgecolors='r', label='relevance vectors')


plt.xlabel('data')
plt.legend(loc="best", scatterpoints=1, prop={'size': 8})
plt.show()
