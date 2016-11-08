# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import pytest
import numpy as np
from pydd.utils import os_utils
from pydd.MLP import MLPfromArray, MLPfromSVM
from sklearn import datasets, metrics, model_selection, preprocessing


def create_dataset(n_classes):
    X, Y = datasets.load_digits(return_X_y=True, n_class=n_classes)
    X = preprocessing.StandardScaler().fit_transform(X)
    return X, Y


# Parameters
seed = 1337
test_size = 0.2
n_classes = 10
connec_param = {'host': 'localhost', 'port': 8081}

# Create dataset
X, Y = create_dataset(n_classes)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y,
                                                                    test_size=test_size,
                                                                    random_state=seed)


class TestSVM(object):
    def test_classification(self):

        params = {'gpu': True, 'nclasses': n_classes}
        params.update(connec_param)

        train_path = os.path.abspath('x_train.svm')
        test_path = os.path.abspath('x_test.svm')
        datasets.dump_svmlight_file(x_train, y_train, train_path)
        datasets.dump_svmlight_file(x_test, y_test, test_path)

        clfs = [
            [{'filepaths': train_path}, test_path, MLPfromSVM(**params)],
            [{'filepaths': [train_path, test_path]}, test_path, MLPfromSVM(**params)],
            [{'X': x_train, 'Y': y_train}, x_test, MLPfromArray(**params)],
            [{'X': x_train, 'Y': y_train, 'validation_data': [(x_test, y_test)]}, x_test, MLPfromArray(**params)]
        ]

        for fit_param, predict_params, clf in clfs:
            clf.fit(**fit_param)
            acc = metrics.accuracy_score(y_test, clf.predict(predict_params))
            assert acc > 0.95

        os_utils._remove_files([train_path, test_path])

    def test_predict_from_model(self):

        snames = ['svm_predict_from_model', 'array_predict_from_model']
        model_repo = [os.path.abspath('model_svm'), os.path.abspath('model_array')]
        params = {'nclasses': n_classes, 'gpu': True}
        params.update(connec_param)

        train_path = os.path.abspath('x_train.svm')
        test_path = os.path.abspath('x_test.svm')
        datasets.dump_svmlight_file(x_train, y_train, train_path)
        datasets.dump_svmlight_file(x_test, y_test, test_path)

        # We make sure model repo does not exist
        for folder in model_repo:
            if os.path.exists(folder):
                os_utils._remove_dirs([folder])
        os_utils._create_dirs(model_repo)

        # Create model, make sure the sname is not used by the server
        clf_svm = MLPfromSVM(sname=snames[0], repository=model_repo[0], **params)
        clf_array = MLPfromArray(sname=snames[1], repository=model_repo[1], **params)

        clf_svm.fit([train_path, test_path], iterations=300)
        clf_array.fit(x_train, y_train, validation_data=[(x_test, y_test)], iterations=300)

        y_pred_svm = clf_svm.predict(test_path)
        y_pred_array = clf_array.predict(x_test)

        # Load from existing model
        params = {'nclasses': n_classes, 'finetuning': True, 'template': None}
        params.update(connec_param)
        clf_svm = MLPfromSVM(sname=snames[0], repository=model_repo[0], **params)
        clf_array = MLPfromArray(sname=snames[1], repository=model_repo[1], **params)

        assert np.array_equal(y_pred_svm, clf_svm.predict(test_path))
        assert np.array_equal(y_pred_array, clf_array.predict(x_test))
        os_utils._remove_files([train_path, test_path])
        os_utils._remove_dirs(model_repo)

if __name__ == '__main__':
    pytest.main([__file__])
