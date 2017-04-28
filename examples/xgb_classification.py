# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:

Make sure DeepDetect is compiled with DUSE_XGBOOST ON

"""

import os
import numpy as np
from pydd.models import XGB
from pydd.connectors import SVMConnector
from sklearn import datasets, model_selection, preprocessing, metrics

# Parameters
seed = 1337
n_samples = 10000
n_features = 20
n_classes = 2
host = 'localhost'
port = 8080
np.random.seed(seed)  # for reproducibility
split_params = {'test_size': 0.2, 'random_state': seed}
booster_params = {"max_depth": 10, "subsample": 0.8, "eta": 0.3}


# create dataset
X, y = datasets.make_classification(n_samples=n_samples, class_sep=0.4, n_features=n_features, n_classes=n_classes, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

# store dataset
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)


# train model
train_data, val_data = SVMConnector(train_path), SVMConnector(test_path)

clf = XGB(host=host, port=port, nclasses=n_classes)
clf.fit(train_data, validation_data=[val_data], **booster_params)

# predict/metrics
y_test_prob = clf.predict_proba(test_path)
y_test_pred = y_test_prob.argmax(-1)
print(metrics.classification_report(y_test, y_test_pred))
