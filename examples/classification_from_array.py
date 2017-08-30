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

split_params = {"test_size": 0.2, "random_state": seed}
model_params = {"port": 8080, "nclasses": n_classes, "gpu": True}
solver_params = {'iterations': 100, 'solver_type': "SGD",
                 'base_lr': 0.01, 'gamma': 0.1, 'stepsize': 30, 'momentum': 0.9}
# class weight value for each class
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1]


# create dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, **split_params)

# Define models and solver
clf = MLP(**model_params)
solver = GenericSolver(**solver_params)

train_data, test_data = ArrayConnector(xtr, ytr), ArrayConnector(xte, yte)

logs = clf.fit(train_data, validation_data=[test_data], solver=solver, class_weights=class_weights, batch_size=128)
yte_pred = clf.predict(test_data)
report = metrics.classification_report(yte, yte_pred)
print(report)
