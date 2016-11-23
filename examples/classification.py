# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pydd.MLP import MLPfromArray
from sklearn import datasets, metrics, model_selection, preprocessing


# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
params = {'port': 8085, 'nclasses': n_classes, 'gpu': True}
split_params = {'test_size': 0.2, 'random_state': seed}

# Arbitrary list of class weights to asses model behavior
class_weights = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1],
                 [2., 1., 1., 1., 1., 1., 1., 1., 1., 1],
                 [4., 1., 1., 1., 1., 1., 1., 1., 1., 1],
                 [100., 1., 1., 1., 1., 1., 1., 1., 1., 1],
                 ]
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

for cw in class_weights:
    clf = MLPfromArray(**params)
    clf.fit(x_train, y_train, iterations=500, batch_size=128, class_weights=cw,weight_decay=0.000001)
    y_pred = clf.predict(x_test)
    report = metrics.classification_report(y_test, y_pred)
    print(report)
