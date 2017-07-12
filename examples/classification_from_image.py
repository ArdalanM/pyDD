# -*- coding: utf-8 -*-
"""
@author: Yassine BEZZA <bezzayassine@gmail.com>
@brief:
"""
import os
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import ImageConnector, LMDBConnector
from pydd.utils import os_utils
from sklearn import metrics, model_selection
from sklearn.datasets import fetch_mldata
from scipy.misc import imsave
from urllib.request import urlretrieve
import lmdb

#Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10

# Load dataset
data_home = os.path.abspath('mnist_data')
while True:
    try:
        mnist = fetch_mldata('MNIST original', data_home=data_home)
    except:
        continue
    break
X, y = mnist.data, mnist.target.astype(np.int)

# Split data in train and test
split_params = {"test_size": 0.2, "random_state": seed}
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, **split_params)

## Create dataset images
# Train
train_dir = os.path.abspath('train_images')
dict_uri_train = {}
try:
    os.mkdir(train_dir)   
except:
    pass
list_labels = list(set(y))
for label in list_labels:
    try:
        os.mkdir(os.path.join(train_dir, str(label)))
    except:
        pass

for i in range(xtr.shape[0]):
    filename = os.path.join(train_dir, str(ytr[i]), str(i) + '.jpeg')
    imsave(filename, xtr[i].reshape((28, 28)))
    dict_uri_train[filename] = i

# Test
test_dir = os.path.abspath('test_images')
dict_uri = {}
try:
    os.mkdir(test_dir)   
except:
    pass
for i in range(xte.shape[0]):
    filename = os.path.join(test_dir, str(i) + '.jpeg')
    imsave(filename, xte[i].reshape((28, 28)))
    dict_uri[filename] = i

# Create repistory for model
model_dir = os.path.abspath('lenet')
try:
    os.mkdir(model_dir)   
except:
    pass
links = [
    'https://raw.githubusercontent.com/beniz/deepdetect/master/examples/caffe/mnist/lenet_deploy.prototxt',
    'https://raw.githubusercontent.com/beniz/deepdetect/master/examples/caffe/mnist/lenet_solver.prototxt',
    'https://raw.githubusercontent.com/beniz/deepdetect/master/examples/caffe/mnist/lenet_train_test.prototxt'
]
try:
    for link in links:
        filename = os.path.join(model_dir, link.split('/')[-1])
        urlretrieve(link, filename)
except Exception as why:
    print('Fail downloading or saving the file ', link)

# Create service and train
params = {"port": 8080, "nclasses": n_classes, "gpu": True, "repository": model_dir, "template": None}
data = ImageConnector(path=train_dir, bw=True, shuffle=True, width=28, height=28)
params.update({'connector': data})
solver = GenericSolver()
clf = MLP(**params)

logs = clf.fit(data,  solver=solver, batch_size=32, metrics=['acc', 'mcll', 'f1'])

# Prediction from image folder
test_data = ImageConnector(path=test_dir, bw=True, shuffle=False, width=28, height=28)
yte_pred = clf.predict(test_data, batch_size=32, dict_uri=dict_uri)
report = metrics.classification_report(yte, yte_pred)   
print(report)

# Prediction from lmdb is possible
# Find a way to get
lmdb_dir = os.path.join(model_dir, 'test.lmdb')
test_data = LMDBConnector(path=lmdb_dir)
yval_pred = clf.predict(test_data, batch_size=32)

# Open lmdb file
lmdb_env = lmdb.open(lmdb_dir, lock=False, readonly=True)
nb_examples = lmdb_env.stat()['entries']
yval = np.zeros((nb_examples, ), dtype=np.int)

# Read lmdb
with lmdb_env.begin() as txn:
    txn_curs = txn.cursor()
    for key, value in txn_curs:
        # Get true label and order of insertion
        row_number = int(key.decode("utf-8").split('_')[0])
        true_label = int(key.decode("utf-8").split('/')[-2])
        yval[row_number] = true_label

report = metrics.classification_report(yval, yval_pred)
print(report)

# Remove dumped files
os_utils._remove_dirs([test_dir, train_dir, model_dir, data_home])
