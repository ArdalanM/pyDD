# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import pytest
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from pydd.utils import os_utils
from pydd.models import MLP
from pydd.solver import GenericSolver
from pydd.connectors import ArrayConnector, SVMConnector
from sklearn import datasets, metrics, model_selection, preprocessing

# Parameters
seed = 1337
test_size = 0.2
n_classes = 10
connec_param = {'host': 'localhost', 'port': 8085}

# Create dataset
X, Y = datasets.load_digits(return_X_y=True, n_class=n_classes)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
tr_f = os.path.abspath('x_train.svm')
te_f = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(xtr, ytr, tr_f)
datasets.dump_svmlight_file(xte, yte, te_f)


class TestSVM(object):
    def test_classification(self):

        params = {'gpu': True, 'nclasses': n_classes}
        params.update(connec_param)
        solver = GenericSolver(iterations=100, base_lr=0.01, gamma=0.5, stepsize=10, momentum=0.9)

        clfs = [
            # no validation set
            [ArrayConnector(xtr, ytr), [], MLP(**params)],
            [SVMConnector([tr_f]), [], MLP(**params)],

            # with validation set
            [SVMConnector([tr_f]), [SVMConnector([tr_f])], MLP(**params)],
            [ArrayConnector(xtr, ytr), [ArrayConnector(xte, yte)], MLP(**params)],

            # sparse
            [ArrayConnector(csc_matrix(xtr), ytr), [], MLP(**params)],
            [ArrayConnector(csc_matrix(xtr), ytr), [ArrayConnector(csc_matrix(xte), yte)], MLP(**params)],
        ]

        for tr_data, te_data, clf in clfs:
            clf.fit(tr_data, te_data, solver)
            y_pred = clf.predict(ArrayConnector(xtr))
            acc = metrics.accuracy_score(ytr, y_pred)
            assert acc > 0.7

    def test_predict_from_model_svm(self):
        # TODO: prediction not equal to label when loading model
        params = {'nclasses': n_classes, 'gpu': True}
        params.update(connec_param)

        tr_data = SVMConnector(paths=[tr_f])
        te_data = SVMConnector(paths=[te_f])

        # Create model, make sure the sname is not used by the server
        clf = MLP(**params)
        solver = GenericSolver(iterations=50, base_lr=0.05, gamma=0.1, stepsize=10, momentum=0.9)
        clf.fit(tr_data, validation_data=[te_data], solver=solver)
        y_pred_tr = clf.predict(tr_data)
        y_pred_te = clf.predict(te_data)

        # Load from existing model
        params = {'nclasses': n_classes, 'finetuning': True, 'template': None}
        params.update(connec_param)
        clf = MLP(sname=clf.sname, repository=clf.model['repository'], **params)
        # assert np.array_equal(y_pred_tr, clf.predict(tr_data))
        # assert np.array_equal(y_pred_te, clf.predict(te_data))

    def test_predict_from_model_array(self):

        params = {'nclasses': n_classes, 'gpu': True}
        params.update(connec_param)

        tr_data = ArrayConnector(xtr, ytr)
        te_data = ArrayConnector(xte, yte)

        # Create model, make sure the sname is not used by the server
        clf = MLP(**params)
        solver = GenericSolver(iterations=100, base_lr=0.01, gamma=0.1, stepsize=10, momentum=0.9)
        clf.fit(tr_data, validation_data=[te_data], solver=solver)
        y_pred_tr = clf.predict(tr_data)
        y_pred_te = clf.predict(te_data)

        # Load from existing model
        params = {'nclasses': n_classes, 'finetuning': True, 'template': None}
        params.update(connec_param)
        clf = MLP(sname=clf.sname, repository=clf.model['repository'], **params)

        assert np.array_equal(y_pred_tr, clf.predict(tr_data))
        assert np.array_equal(y_pred_te, clf.predict(te_data))

    def test_sklearn(self):
        # TODO: make models compatible with sklearn
        pass
        # params = {'gpu': True, 'nclasses': n_classes}
        # params.update(connec_param)
        # param_grid = {'layers': [[10], [100]]}
        #
        # skf = model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        # scorer = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
        #
        # clf = MLP(**params)
        # cv_scores = model_selection.cross_val_score(clf, xtr, ytr, scoring=scorer, cv=skf)
        #
        # for clf in clfs:
        #     cv_scores = model_selection.cross_val_score(clf, X, Y, scoring=scorer, cv=skf)
        #
        #     y_pred = model_selection.cross_val_predict(clf, X, Y, cv=skf, method='predict_proba')
        #     score = metrics.accuracy_score(Y, y_pred.argmax(-1))
        #
        #     grid = model_selection.GridSearchCV(clf, param_grid, scoring=scorer, cv=skf)
        #     grid.fit(X, Y)
        #     grid_best_score = grid.best_score_
        #
        #     assert score and cv_scores.all() and grid_best_score > 0.9


if __name__ == '__main__':
    pytest.main([__file__])
