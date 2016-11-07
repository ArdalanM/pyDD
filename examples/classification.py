# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
from sklearn import datasets, model_selection, preprocessing, metrics
from pydd.MLP import MLPfromSVM, MLPfromArray
from pydd.utils import os_utils


def create_dataset():

    X, Y = datasets.load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2,
                                                                        random_state=1337)
    scl = preprocessing.StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    return x_train, y_train, x_test, y_test

# Parameters
seed = 1337
nclasses = 10
test_size = 0.2
params = {'port': 8081, 'nclasses': nclasses, 'layers': [100, 100], 'activation': 'relu',
          'dropout': 0.5, 'db': True, 'gpu': True}

# Create dataset
x_train, y_train, x_test, y_test = create_dataset()
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)


clf_svm = MLPfromSVM(**params)
clf_array = MLPfromArray(**params)

clf_svm.fit([train_path, test_path], iterations=1000, test_interval=10)
clf_array.fit(x_train, y_train, validation_data=[(x_test, y_test)], iterations=1000, test_interval=10)


for X, clf in [[test_path,  clf_svm], [x_test, clf_array]]:
    y_prob = clf.predict_proba(X)
    y_pred = clf.predict(X)

    print("Model: {} Loss: {}".format(clf, metrics.log_loss(y_test, y_prob)))
    print("Model: {} Accuracy: {}".format(clf, metrics.accuracy_score(y_test, y_pred)))
os_utils._remove_files([train_path, test_path])

