# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
from sklearn import datasets, preprocessing, model_selection, metrics
from pydd.MLP import MLPfromSVM, MLPfromArray
from pydd.utils import os_utils


def create_dataset():

    X, Y = datasets.load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2,
                                                                        random_state=1337)
    scl = preprocessing.StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    return x_train, y_train, x_test, y_test


# Parameters
snames = ['svm_predict_from_model', 'array_predict_from_model']
model_repo = [os.path.abspath('model_svm'), os.path.abspath('model_array')]
params = {'host': 'localhost', 'port': 8081, 'nclasses': 10, 'layers': [100, 100], 'activation': 'relu',
          'dropout': 0.5, 'db': True, 'gpu': True}

# We make sure model repo does not exist
for folder in model_repo:
    if os.path.exists(folder):
        os_utils._remove_dirs([folder])
os_utils._create_dirs(model_repo)

# Create dataset
x_train, y_train, x_test, y_test = create_dataset()
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)

# Create model, make sure the sname is not used by the server
clf_svm = MLPfromSVM(sname=snames[0], repository=model_repo[0], **params)
clf_array = MLPfromArray(sname=snames[1], repository=model_repo[1], **params)

clf_svm.fit([train_path, test_path], iterations=1000, test_interval=10)
clf_array.fit(x_train, y_train, validation_data=[(x_test, y_test)], iterations=1000, test_interval=10)

del clf_svm
del clf_array

# Load from existing model
params = {'host': 'localhost', 'port': 8081, 'nclasses': 10, 'finetuning': True, 'template': None}
clf_svm = MLPfromSVM(sname=snames[0], repository=model_repo[0], **params)
clf_array = MLPfromArray(sname=snames[1], repository=model_repo[1], **params)


for X, clf in [[test_path,  clf_svm], [x_test, clf_array]]:
    y_prob = clf.predict_proba(X)
    y_pred = clf.predict(X)
    print("Model: {} Loss: {}".format(clf, metrics.log_loss(y_test, y_prob)))
    print("Model: {} Accuracy: {}".format(clf, metrics.accuracy_score(y_test, y_pred)))

os_utils._remove_files([train_path, test_path])
os_utils._remove_dirs(model_repo)
