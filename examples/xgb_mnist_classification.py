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
n_classes = 2
params = {'port': 8085, 'nclasses': n_classes}
split_params = {'test_size': 0.2, 'random_state': seed}

X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)

st = time.time()
clf = XGBClassifier(**params)
clf.fit([train_path, test_path], iterations=1, max_depth=20, tree_method='hist',
        grow_policy="depthwise", scale_pos_weight=10)

y_test_prob = clf.predict_proba(test_path)
y_test_pred = y_test_prob.argmax(-1)

print(metrics.classification_report(y_test, y_test_pred))
print(time.time()-st)