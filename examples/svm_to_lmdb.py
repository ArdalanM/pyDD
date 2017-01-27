# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
from sklearn import datasets, preprocessing, model_selection
from pydd.utils import lmdb_utils
from pydd.MLP import MLPfromSVM
from pydd.LR import LRfromSVM

# Create dataset
X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1337)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)


train_lmdb_path = lmdb_utils.create_lmdb_from_svm(train_path, os.path.abspath('train.lmdb'), port=8082, gpu=True)
test_lmdb_path = lmdb_utils.create_lmdb_from_svm(test_path, os.path.abspath('test.lmdb'), port=8082, gpu=True)

# Training model from lmdb data
clf = MLPfromSVM(port=8082, nclasses=10, gpu=True)
clf.fit([train_path, test_path], lmdb_paths=[train_lmdb_path, test_lmdb_path])
y = clf.predict_proba(train_path)