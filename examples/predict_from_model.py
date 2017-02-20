# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
from pydd.utils import os_utils
from pydd.models import MLPfromSVM, MLPfromArray
from sklearn import datasets, preprocessing, model_selection, metrics


# Parameters
snames = ['svm_predict_from_model', 'array_predict_from_model']
model_repo = [os.path.abspath('model_svm'), os.path.abspath('model_array')]
params = {'host': 'localhost', 'port': 8085,
          'nclasses': 10, 'layers': [100, 100],
          'activation': 'relu', 'dropout': 0.2,
          'db': True, 'gpu': True}

# We make sure model repo does not exist
for folder in model_repo:
    if os.path.exists(folder):
        os_utils._remove_dirs([folder])
os_utils._create_dirs(model_repo)


# Create dataset
X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1337)

# Save data in .svm format
train_path = os.path.abspath('x_train.svm')
test_path = os.path.abspath('x_test.svm')
datasets.dump_svmlight_file(x_train, y_train, train_path)
datasets.dump_svmlight_file(x_test, y_test, test_path)

# Create model, make sure the sname is not used by the server
clf_svm = MLPfromSVM(repository="/data/sharedData/prod-affinitas/models/dd_models/test_yassine", port=8082, nclasses=2,
                     finetuning=True, template=None, gpu=True)
clf_svm.predict_proba("/data/ardalan.mehrani/ioSquare/prod-affinitas/data/output/01-02-full_c@31551_y@2_seed@1337/val_r@1318499_c@31551_y@2_seed@1337.svm")


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
    print('-' * 50)
    print("Model: {}".format(clf))
    print("Model: Accuracy: {}, Loss: {}".format(metrics.accuracy_score(y_test, y_pred),
                                                 metrics.log_loss(y_test, y_prob)))

os_utils._remove_files([train_path, test_path])
os_utils._remove_dirs(model_repo)
