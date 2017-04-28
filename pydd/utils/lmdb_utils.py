# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import shutil
import tempfile
from pydd.models import MLP
from pydd.solver import GenericSolver
from pydd.connectors import SVMConnector


def create_lmdb_from_svm(svm_path, lmdb_path, vocab_path=None, host='localhost',
                         port=8085, nclasses=2, gpu=True, tmp_folder=None):

    if os.path.exists(lmdb_path):
        print("warning: {} exist, overwriting it".format(lmdb_path))

    tmp_folder = tempfile.mkdtemp(prefix="pydd_", dir=tmp_folder) if tmp_folder else tempfile.mkdtemp(prefix="pydd_")

    train_data = SVMConnector(path=svm_path)
    optimizer = GenericSolver(solver_type='SGD', base_lr=0.01, iterations=1)

    clf = MLP(host=host, port=port, nclasses=nclasses, gpu=gpu, repository=tmp_folder)
    clf.fit(train_data, solver=optimizer)

    shutil.move(os.path.join(tmp_folder, "train.lmdb"), lmdb_path)
    if vocab_path:
        shutil.move(os.path.join(tmp_folder, "vocab.dat"), vocab_path)

    # delete service
    clf.delete_service(clf.sname, clear='lib')

    # delete tmp_folder
    shutil.rmtree(tmp_folder)

    return lmdb_path, vocab_path


if __name__ == "__main__":

    from sklearn import datasets, preprocessing, model_selection

    # parameters
    port = 8085
    gpu = False
    nclasses = 10
    seed = 1337
    test_size = 0.2
    params = {'port': 8085, 'nclasses': nclasses, 'gpu': gpu}

    # Create dataset
    X, Y = datasets.load_digits(n_class=params['nclasses'], return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    # Save data in .svm format
    tr_svm_f, tr_lmdb_f = os.path.abspath('x_train.svm'), os.path.abspath('x_train.lmdb')
    te_svm_f, te_lmdb_f = os.path.abspath('x_test.svm'), os.path.abspath('x_test.lmdb')
    vocab_path = os.path.abspath('vocab.dat')

    datasets.dump_svmlight_file(x_train, y_train, tr_svm_f)
    datasets.dump_svmlight_file(x_test, y_test, te_svm_f)

    # create lmdb and vocab file
    create_lmdb_from_svm(svm_path=tr_svm_f, lmdb_path=tr_lmdb_f, vocab_path=vocab_path, **params)
    create_lmdb_from_svm(svm_path=te_svm_f, lmdb_path=te_lmdb_f, **params)

    tr_data = SVMConnector(path=tr_svm_f, lmdb_path=tr_lmdb_f, vocab_path=vocab_path)
    te_data = SVMConnector(path=tr_svm_f, lmdb_path=tr_lmdb_f)

    optimizer = GenericSolver(solver_type='SGD', base_lr=0.01, iterations=100)
    clf = MLP(**params)
    clf.fit(tr_data, validation_data=[te_data], solver=optimizer)

    y_pred_lmdb = clf.predict_proba(te_data)


