# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import time
import numpy as np
from pydd.XGB import XGBClassifier
from sklearn import datasets, model_selection, preprocessing, metrics

# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility

n_samples = 10000
n_features = 20
n_classes = 2

split_params = {'test_size': 0.2, 'random_state': seed}

X, y = datasets.make_classification(n_samples=n_samples, class_sep=0.4, n_features=n_features, n_classes=n_classes, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)

clf = XGBClassifier(port=8085, nclasses=n_classes)

booster_params = {"max_depth": 10, "subsample": 0.8, "eta": 0.3, "drop_rate": 0.4, "skip_drop": 0.4}
clf.fit([train_path, test_path], booster="dart", iterations=20, **booster_params)

y_test_prob = clf.predict_proba(test_path)
y_test_pred = y_test_prob.argmax(-1)

print(metrics.classification_report(y_test, y_test_pred))
