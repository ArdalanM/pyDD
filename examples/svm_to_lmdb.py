# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
from sklearn import datasets, preprocessing, model_selection, metrics
from pydd.utils import lmdb_utils
from pydd.MLP import MLPfromSVM

host = "localhost"
port = 8085
gpu = True
nclasses = 10

# Create dataset
X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1337)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)


# Create lmdb dataset
train_lmdb_path = os.path.abspath('train.lmdb')
test_lmdb_path = os.path.abspath('test.lmdb')
vocab_path = os.path.abspath('vocab.dat')
if not os.path.isdir(os.path.abspath('train.lmdb')):
    train_lmdb_path, vocab_path = lmdb_utils.create_lmdb_from_svm(train_path, train_lmdb_path, vocab_path,
                                                                  port=port, gpu=gpu)

if not os.path.isdir(os.path.abspath('test.lmdb')):
    test_lmdb_path, _ = lmdb_utils.create_lmdb_from_svm(test_path, test_lmdb_path, port=port, gpu=gpu)

# Training model from lmdb data
clf = MLPfromSVM(port=port, nclasses=nclasses, gpu=gpu)
clf.fit([train_path, test_path], lmdb_paths=[train_lmdb_path, test_lmdb_path], vocab_path=vocab_path,
        iterations=500, solver_type='SGD', test_interval=100)

y_pred = clf.predict(test_path)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
