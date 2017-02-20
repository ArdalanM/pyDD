# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import ArrayConnector
from sklearn import datasets, metrics, model_selection, preprocessing

# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
params = {"port": 8085, "nclasses": n_classes, "gpu": True}
split_params = {"test_size": 0.2, "random_state": seed}

# Define dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)


# Define models and class weights
clfs = [MLP(**params)]
solver = GenericSolver(iterations=100, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)

class_weights = [
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1],
    [2., 1., 1., 1., 1., 1., 1., 1., 1., 1],
    [4., 1., 1., 1., 1., 1., 1., 1., 1., 1],
    [100., 1., 1., 1., 1., 1., 1., 1., 1., 1]
]

# Define connectors
train_data = ArrayConnector(x_train, y_train)
test_data = ArrayConnector(x_test, y_test)


for clf in clfs:
    for cw in class_weights:
        print('-' * 50)
        print(clf, cw)
        logs = clf.fit(train_data, solver=solver, class_weights=cw, batch_size=128)
        y_pred = clf.predict(test_data)
        report = metrics.classification_report(y_test, y_pred)
        print(report)
