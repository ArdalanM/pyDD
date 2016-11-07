# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from sklearn import datasets, metrics, model_selection
from pydd.MLP import MLPfromArray

# Parameters
seed = 1337
params = {'port': 8081, 'nclasses': 3}
class_weights = [1., 1., 1.]
test_size = 0.2

np.random.seed(seed)  # for reproducibility
X, y = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

clf = MLPfromArray(**params)
clf.fit(x_train, y_train, iterations=200, batch_size=150, class_weights=class_weights)

y_prob = clf.predict_proba(x_test)
y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Log loss: ", metrics.log_loss(y_test, y_prob))
print(np.bincount(y_pred.ravel()))
