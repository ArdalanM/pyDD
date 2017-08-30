# -*- coding: utf-8 -*-
"""
@author: Yassine BEZZA <bezzayassine@gmail.com>
@brief:
"""
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import ArrayConnector
from sklearn import datasets, metrics, model_selection, preprocessing
import subprocess
import sys

# Redirect stdout to a file
orig_stdout = sys.stdout
log_file = 'logs.txt'
f = open(log_file, 'w')
sys.stdout = f

# Parameters
seed = 1337
np.random.seed(seed)  # for reproducibility
n_classes = 10
model_params = {"port": 8080, "nclasses": n_classes, "gpu": True}
split_params = {"test_size": 0.2, "random_state": seed}
solver_params = {'iterations': 5000, 'solver_type': "SGD", 'base_lr': 0.01, 'gamma': 0.1, 'stepsize': 30, 'momentum': 0.9}
class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1]

# create dataset
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, **split_params)

# Define models and class weights
clf = MLP(**model_params)

solver = GenericSolver(**solver_params)
# one class weight value for each class

train_data, test_data = ArrayConnector(xtr, ytr), ArrayConnector(xte, yte)

# Start visdom on another process
#start_visdom = ["python'",  "-m",  "visdom.server"]
#subprocess.Popen(["python", "-m", "visdom.server"])

# Listen visdom on logs file
#listen_logs = ["pydd",  "--log_dir", log_file]
#subprocess.Popen(listen_logs)

# Start listening to the port
logs = clf.fit(train_data, validation_data=[test_data], solver=solver, class_weights=class_weights, batch_size=128)
yte_pred = clf.predict(test_data)
report = metrics.classification_report(yte, yte_pred)
print(report)

# Close the output file
sys.stdout = orig_stdout
f.close()
