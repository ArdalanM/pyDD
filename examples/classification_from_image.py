# -*- coding: utf-8 -*-
"""
@author: Yassine BEZZA <bezzayassine@gmail.com>
@brief:
"""
import os
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import ImageConnector
from pydd.utils import os_utils
from sklearn import datasets, metrics
from scipy.misc import imsave
# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
params = {"port": 8080, "nclasses": n_classes, "gpu": True, "template": 'googlenet'}
split_params = {"test_size": 0.2, "random_state": seed}

# create dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = X.astype(np.uint16)

# Create dataset images
img_dir = os.path.abspath('data_imgs')
try:
    os.mkdir(img_dir)
except:
    pass
list_labels = list(set(y))
for label in list_labels:
    try:
        os.mkdir(os.path.join(img_dir, str(label)))
    except:
        pass

for i in range(X.shape[0]):
    imsave(os.path.join(img_dir, str(y[i]), str(i) + '.jpeg'), X[i].reshape((8,8)))

# Define models and class weights
data = ImageConnector(path=img_dir)
params.update({'connector': data})
clf = MLP(**params)

solver = GenericSolver(iterations=500, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)
# one class weight value for each class
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
logs = clf.fit(data,  solver=solver, class_weights=class_weights, batch_size=8)

# TO do make a dataset for prediction as well
#yte_pred = clf.predict(test_data)
#report = metrics.classification_report(yte, yte_pred)
#print(report)
