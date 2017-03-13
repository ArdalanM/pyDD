# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import pytest
import numpy as np
from scipy.sparse import csc_matrix
from pydd.utils import os_utils, lmdb_utils
from pydd.models import MLP, LR
from pydd.solver import GenericSolver
from pydd.connectors import ArrayConnector, SVMConnector
from sklearn import datasets, metrics, model_selection, preprocessing

##############
# parameters #
##############
seed = 1337
test_size = 0.2
n_classes = 10

# dd params
nn_params = {'host': 'localhost', 'port': 8085, 'gpu': True}
solver_param = {"iterations": 100, "base_lr": 0.01, "gamma": 0.1, "stepsize": 30, "momentum": 0.9}
# xgb_params = {'host': 'localhost', 'port': 8085}
# booster_params = {"max_depth": 10, "subsample": 0.8, "eta": 0.3}

##################
# create dataset #
##################
X, Y = datasets.load_digits(return_X_y=True, n_class=n_classes)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
tr_f = os.path.abspath('x_train.svm')
te_f = os.path.abspath('x_test.svm')

#####################
# create connectors #
#####################
# array connector
xtr_arr, xte_arr = ArrayConnector(xtr, ytr), ArrayConnector(xte, yte)

# svm connector
xtr_svm, xte_svm = SVMConnector(tr_f), SVMConnector(te_f)

# array sparse connector
xtr_sparse, xte_sparse = ArrayConnector(csc_matrix(xtr), ytr), ArrayConnector(csc_matrix(xte), yte)


class TestSVM(object):
    def test_classification(self):

        params = nn_params.copy()
        params.update({'nclasses': n_classes})
        optimizer = GenericSolver(**solver_param)
        datasets.dump_svmlight_file(xtr, ytr, tr_f)
        datasets.dump_svmlight_file(xte, yte, te_f)

        clfs = [
            # array connector without validation set
            [xtr_arr, [], MLP(**params)],
            [xtr_arr, [], LR(**params)],

            # sparse array connector without validation set
            [xtr_sparse, [], MLP(**params)],
            [xtr_sparse, [], LR(**params)],

            # svm connector without validation set
            [xtr_svm, [], MLP(**params)],
            [xtr_svm, [], LR(**params)],

            # array connector with validation set
            [xtr_arr, [xte_arr], MLP(**params)],
            [xtr_arr, [xte_arr], LR(**params)],

            # svm connector with validation set
            [xtr_svm, [xte_svm], MLP(**params)],
            [xtr_svm, [xte_svm], LR(**params)],
        ]

        for tr_data, te_data, clf in clfs:
            clf.fit(tr_data, te_data, optimizer)
            y_pred = clf.predict(tr_data)
            acc = metrics.accuracy_score(ytr, y_pred)
            assert acc > 0.7

        os_utils._remove_files([tr_f, te_f])

    def test_predict_from_model_svm(self):

        params = nn_params.copy()
        params.update({'nclasses': n_classes})
        optimizer = GenericSolver(**solver_param)
        datasets.dump_svmlight_file(xtr, ytr, tr_f)
        datasets.dump_svmlight_file(xte, yte, te_f)

        # Train model
        clf = MLP(**params)
        clf.fit(xtr_svm, validation_data=[xte_svm], solver=optimizer)
        y_pred_tr = clf.predict(xtr_svm)
        y_pred_te = clf.predict(xte_svm)

        # Load from tained model
        params = nn_params.copy()
        params.update({'finetuning': True, 'template': None, 'nclasses': n_classes})
        clf = MLP(sname=clf.sname, repository=clf.model['repository'], **params)

        assert np.array_equal(y_pred_tr, clf.predict(xtr_svm))
        assert np.array_equal(y_pred_te, clf.predict(xte_svm))
        os_utils._remove_files([tr_f, te_f])

    def test_predict_from_model_array(self):

        params = nn_params.copy()
        params.update({'nclasses': n_classes})
        optimizer = GenericSolver(**solver_param)
        datasets.dump_svmlight_file(xtr, ytr, tr_f)
        datasets.dump_svmlight_file(xte, yte, te_f)

        # Train model
        clf = MLP(**params)
        clf.fit(xtr_arr, validation_data=[xte_arr], solver=optimizer)
        y_pred_tr = clf.predict(xtr_arr)
        y_pred_te = clf.predict(xte_arr)

        # Load from tained model
        params = nn_params.copy()
        params.update({'finetuning': True, 'template': None, 'nclasses': n_classes})
        clf = MLP(sname=clf.sname, repository=clf.model['repository'], **params)

        assert np.array_equal(y_pred_tr, clf.predict(xtr_arr))
        assert np.array_equal(y_pred_te, clf.predict(xte_arr))
        os_utils._remove_files([tr_f, te_f])

    def test_lmdb_creation(self):

        params = nn_params.copy()
        params.update({'nclasses': n_classes})

        # Create dataset
        X, Y = datasets.load_digits(return_X_y=True)
        X = preprocessing.StandardScaler().fit_transform(X)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

        # Save data in .svm format
        tr_svm_f, tr_lmdb_f = os.path.abspath('x_train.svm'), os.path.abspath('x_train.lmdb')
        te_svm_f, te_lmdb_f = os.path.abspath('x_test.svm'), os.path.abspath('x_test.lmdb')
        vocab_path = os.path.abspath('vocab.dat')

        datasets.dump_svmlight_file(x_train, y_train, tr_svm_f)
        datasets.dump_svmlight_file(x_test, y_test, te_svm_f)

        lmdb_utils.create_lmdb_from_svm(svm_path=tr_svm_f, lmdb_path=tr_lmdb_f, vocab_path=vocab_path, **params)
        lmdb_utils.create_lmdb_from_svm(svm_path=te_svm_f, lmdb_path=te_lmdb_f, **params)

        tr_lmdb = SVMConnector(path=tr_svm_f, lmdb_path=tr_lmdb_f, vocab_path=vocab_path)
        te_lmdb = SVMConnector(path=te_svm_f, lmdb_path=te_lmdb_f)

        optimizer = GenericSolver(solver_type='SGD', base_lr=0.01, iterations=100)
        clf = MLP(**params)
        clf.fit(tr_lmdb, validation_data=[te_lmdb], solver=optimizer)

        ytr_prob = clf.predict_proba(tr_lmdb)
        acc = metrics.accuracy_score(y_train, ytr_prob.argmax(-1))
        assert acc > 0.7

        os_utils._remove_files([tr_svm_f, te_svm_f, vocab_path])
        os_utils._remove_dirs([tr_lmdb_f, te_lmdb_f])

if __name__ == '__main__':
    pytest.main([__file__])
