# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import SVMConnector
from pydd.utils import os_utils, lmdb_utils
from sklearn import datasets, metrics, model_selection, preprocessing


# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
params = {"port": 8080, "nclasses": n_classes, "gpu": False}
split_params = {"test_size": 0.2, "random_state": seed}

folder = "train-from-lmdb"
if os.path.exists(folder):
    os_utils._remove_dirs([folder])
os_utils._create_dirs([folder])


# create dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, **split_params)

# create and save train.svm and test.svm
tr_f = os.path.abspath("{}/x_train.svm".format(folder))
te_f = os.path.abspath("{}/x_test.svm".format(folder))
datasets.dump_svmlight_file(xtr, ytr, tr_f)
datasets.dump_svmlight_file(xte, yte, te_f)

# create lmdb dataset
tr_lmdb = os.path.abspath("{}/train.lmdb".format(folder))
te_lmdb = os.path.abspath("{}/test.lmdb".format(folder))
vocab_path = os.path.abspath("{}/vocab.dat".format(folder))
lmdb_utils.create_lmdb_from_svm(tr_f, tr_lmdb, vocab_path, **params)
lmdb_utils.create_lmdb_from_svm(te_f, te_lmdb, **params)


# create lmdb connectors
train_data = SVMConnector(path=tr_f, lmdb_path=tr_lmdb, vocab_path=vocab_path)
test_data = SVMConnector(path=te_f, lmdb_path=te_lmdb)


# Training model from lmdb data
clf = MLP(**params)
optimizer = GenericSolver(solver_type='SGD', iterations=500, base_lr=0.01)
logs = clf.fit(train_data, validation_data=[test_data], solver=optimizer)

yte_pred = clf.predict(test_data)
report = metrics.classification_report(yte, yte_pred)
print(report)

os_utils._remove_dirs([folder])
