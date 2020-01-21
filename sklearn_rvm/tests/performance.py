# import timeit
#
#
# # Time classification with the iris dataset.
#
# setup_c = """
# from sklearn.datasets import load_iris
# from sklearn_rvm import EMRVC
# iris = load_iris()
# X = iris.data
# y = iris.target
# clf = EMRVC()
# """
#
# time = timeit.timeit("clf.fit(X, y)", setup=setup_c, number=10)
#
# print("10 runs of Iris classification fitting took {} seconds.".format(time))
#
# # Time regression with the boston ds.
#
# setup_r = """
# from sklearn.datasets import load_boston
# from sklearn_rvm import EMRVR
# boston = load_boston()
# X = boston.data
# y = boston.target
# clf = EMRVR()
# """
#
# time = timeit.timeit("clf.fit(X, y)", setup=setup_r, number=10)
#
# print("10 runs of boston refression fitting took {} seconds.".format(time))