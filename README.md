pyDD: python binding for [DeepDetect](https://github.com/beniz/deepdetect)
============================================================================
<!--[![Build Status](https://travis-ci.org/ArdalanM/pyLightGBM.svg?branch=feat_ci)](https://travis-ci.org/ArdalanM/pyLightGBM)-->
<!--[![Coverage Status](https://coveralls.io/repos/github/ArdalanM/pyLightGBM/badge.svg?branch=master)](https://coveralls.io/github/ArdalanM/pyLightGBM?branch=master)-->
<!--[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)]()-->

TO DO:
 - Support other DeepDetect connectors: `image`, `csv`, `text`
 
Installation
------------
- Install DeepDetect (instruction [here](https://deepdetect.com/overview/installing/)):
- Install pyDD: ```pip install git+https://github.com/ArdalanM/pyDD.git```

Examples
--------
Make sure DeepDetect is up and running:  
`./main/dede`

* Classification from array:  
```python
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import ArrayConnector
from sklearn import datasets, metrics, model_selection, preprocessing

# create dataset
n_classes = 10
X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = model_selection.train_test_split(X, y, test_size=0.2)

# create connector
train_data, test_data = ArrayConnector(xtr, ytr), ArrayConnector(xte, yte)

# Define models and class weights
clf = MLP(port=8085, nclasses=n_classes, gpu=True)
solver = GenericSolver(iterations=10000, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)

logs = clf.fit(train_data, validation_data=[test_data], solver=solver)
yte_pred = clf.predict(test_data)
report = metrics.classification_report(yte, yte_pred)
```

- Classification from svm:  
```python
import numpy as np
from pydd.solver import GenericSolver
from pydd.models import MLP
from pydd.connectors import SVMConnector
from sklearn import datasets, metrics, model_selection, preprocessing

# create connector
n_classes = 10
train_data = SVMConnector(path="x_train.svm")
test_data = SVMConnector(path="x_test.svm")


# Define models and class weights
clf = MLP(port=8085, nclasses=n_classes, gpu=True)
solver = GenericSolver(iterations=10000, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)

logs = clf.fit(train_data, validation_data=[test_data], solver=solver)
yte_pred = clf.predict(test_data)
```

Check out the [example](https://github.com/ArdalanM/pyDD/tree/master/examples) folder for more cases.