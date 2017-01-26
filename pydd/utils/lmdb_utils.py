# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import shutil
import tempfile
from pydd.LR import LRfromSVM


def create_lmdb_from_svm(svm_path, lmdb_path, host='localhost', port=8085, nclasses=2, gpu=True, tmp_folder=None):

    tmp_folder = tempfile.mkdtemp(prefix="pydd_", dir=tmp_folder) if tmp_folder else tempfile.mkdtemp(prefix="pydd_")

    clf = LRfromSVM(host=host, port=port, nclasses=nclasses, gpu=gpu, repository=tmp_folder)
    clf.fit([svm_path], iterations=1)
    shutil.move(os.path.join(tmp_folder, "train.lmdb"), lmdb_path)

    # delete service
    clf.delete_service(clf.sname, clear='lib')

    # delete tmp_folder
    shutil.rmtree(tmp_folder)

    return lmdb_path


if __name__ == "__main__":

    import os
    from sklearn import datasets, preprocessing, model_selection
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

    lmdb_path = create_lmdb_from_svm(train_path, os.path.abspath('train.lmdb'), port=8082, gpu=True)
    lmdb_path1 = create_lmdb_from_svm(test_path, os.path.abspath('test.lmdb'), port=8082, gpu=True)
