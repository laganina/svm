#how to import data

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#svm is a a classifier
#breast cancer data set

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=2)
#C is soft margin
#hard margin is c=0
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

#scoring
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

clf1 = KNeighborsClassifier(n_neighbors=9)
clf1.fit(x_train, y_train)
y_pred1 = clf1.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_pred1)
print(acc1)

#svm is better than knn, tipically knn doesn't work well with huge dimensions
#svm is best bet with acc because of kernel

