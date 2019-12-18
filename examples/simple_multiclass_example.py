from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn_rvm import EMRVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = EMRVC(kernel='linear')
clf.fit(X_train, y_train)

clf.predict(X_test)

print(clf.predict(X_test))
print(clf.predict_proba(X_test))
print(clf.score(X_test, y_test))
