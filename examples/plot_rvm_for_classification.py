#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
RVM for regression
=========================================================
Based on https://github.com/ctgk/PRML/blob/master/notebooks/ch07_Sparse_Kernel_Machines.ipynb
"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn_rvm import EMRVC


def create_toy_data():
    x0 = np.random.normal(size=100).reshape(-1, 2) - 1.
    x1 = np.random.normal(size=100).reshape(-1, 2) + 1.
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int)
    return x, y


x_train, y_train = create_toy_data()

model = EMRVC(kernel='rbf')
model.fit(x_train, y_train)

x0, x1 = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
x = np.array([x0, x1]).reshape(2, -1).T
plt.scatter(x_train[:, 0], x_train[:, 1], s=40, c=y_train, marker='x')
plt.scatter(model.relevance_vectors_[:, 0], model.relevance_vectors_[:, 1], s=100, facecolor='none', edgecolor='g')
plt.contourf(x0, x1, model.predict_proba(x)[:, 1].reshape(100, 100), np.linspace(0, 1, 5), alpha=0.2)
plt.colorbar()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal', adjustable='box')
