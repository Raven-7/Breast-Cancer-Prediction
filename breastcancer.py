

import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import style


cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
#print(x_train,y_train)
classes = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=1)                 #svm algorithm
#clf = KNeighborsClassifier(n_neighbors=13)         #knn algorithm
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test) 

acc = metrics.accuracy_score(y_test,y_pred)
print(acc)

names = ["M", "B"]

for x in range(len(y_pred)):
 print("Predicted: ", names[y_test[x]], "Actual: ", names[y_pred[x]])
