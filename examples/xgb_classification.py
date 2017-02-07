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


params = {'port': 8085, 'nclasses': n_classes}
split_params = {'test_size': 0.2, 'random_state': seed}

X, y = datasets.make_classification(n_samples=n_samples, class_sep=0.4, n_features=n_features, n_classes=n_classes, random_state=seed)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)

st = time.time()
clf = XGBClassifier(**params)
clf.fit([train_path, test_path], iterations=10, max_depth=10,  scale_pos_weight=0.3)

y_test_prob = clf.predict_proba(test_path)
y_test_pred = y_test_prob.argmax(-1)

print(metrics.classification_report(y_test, y_test_pred))
print(time.time()-st)


#              precision    recall  f1-score   support
#
#           0       0.75      0.83      0.79       965
#           1       0.82      0.74      0.78      1035
#
# avg / total       0.79      0.78      0.78      2000
#
# 4.435389757156372
