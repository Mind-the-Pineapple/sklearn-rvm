# #!/usr/bin/python
# # -*- coding: utf-8 -*-
#
# """
# =========================================================
# Simple example precomputed
# ========================================================="""
# print(__doc__)
#
# import numpy as np
# from sklearn.metrics.pairwise import pairwise_kernels
#
# from sklearn_rvm import EMRVC
#
# # General a toy dataset:s it's just a straight line with some Gaussian noise:
# n_samples = 100
# np.random.seed(0)
# X = np.random.normal(size=n_samples)
# y = (X > 0).astype(np.float)
# X[X > 0] *= 4
# X += .3 * np.random.normal(size=n_samples)
#
# X = X[:, np.newaxis]
#
# K = pairwise_kernels(X)
# # Fit the classifier
# clf = EMRVC(kernel="precomputed")
# clf.fit(K, y)
#
# print(clf.predict(K))
# print(clf.predict_proba(K))
# print(clf.score(K, y))
