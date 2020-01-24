#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================================================================
Comparison of relevance vector machine and support vector machine
=====================================================================
"""
print(__doc__)

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn_rvm import EMRVR

np.random.seed(8)
rng = np.random.RandomState(0)

# Generate sample data
X = 4 * np.pi * np.random.random(100) - 2 * np.pi
y = np.sinc(X)
y += 0.25 * (0.5 - rng.rand(X.shape[0]))  # add noise

X = X[:, None]

# Fit SVR
svr = SVR(kernel="rbf", gamma="auto")
stime = time.time()
svr.fit(X, y)
print("Time for SVR fitting: %.3f" % (time.time() - stime))

# Fit RVR
rvr = EMRVR(kernel="rbf")
stime = time.time()
rvr.fit(X, y)
print("Time for RVR fitting: %.3f" % (time.time() - stime))

X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 10000)[:, None]
# Predict using SVR
stime = time.time()
y_svr = svr.predict(X_plot)
print("Time for SVR prediction: %.3f" % (time.time() - stime))

# Predict using Rvm
stime = time.time()
y_rvr = rvr.predict(X_plot, return_std=False)
print("Time for RVR prediction: %.3f" % (time.time() - stime))

stime = time.time()
y_rvr, y_std = rvr.predict(X_plot, return_std=True)
print("Time for RVR prediction with standard-deviation: %.3f" % (time.time() - stime))

# Plot results
fig = plt.figure(figsize=(10, 5))
lw = 2
fig.suptitle("RVR versus SVR", fontsize=16)

plt.subplot(121)
plt.scatter(X, y, marker=".", c="k", label="data")
plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

plt.plot(X_plot, y_svr, color="turquoise", lw=lw, label="SVR")
support_vectors_idx = svr.support_
plt.scatter(X[support_vectors_idx], y[support_vectors_idx], s=80, facecolors="none", edgecolors="r",
            label="support vectors")
plt.ylabel("target")
plt.xlabel("data")
plt.legend(loc="best", scatterpoints=1, prop={"size": 8})

plt.subplot(122)
plt.scatter(X, y, marker=".", c="k", label="data")
plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

plt.plot(X_plot, y_rvr, color="darkorange", lw=lw, label="RVR")
plt.fill_between(X_plot[:, 0], y_rvr - y_std, y_rvr + y_std, color="darkorange", alpha=0.2)
relevance_vectors_idx = rvr.relevance_
plt.scatter(X[relevance_vectors_idx], y[relevance_vectors_idx], s=80, facecolors="none", edgecolors="r",
            label="relevance vectors")

plt.xlabel("data")
plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
plt.show()
