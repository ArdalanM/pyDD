pyDD: python binding for [DeepDetect](https://github.com/beniz/deepdetect)
============================================================================
<!--[![Build Status](https://travis-ci.org/ArdalanM/pyLightGBM.svg?branch=feat_ci)](https://travis-ci.org/ArdalanM/pyLightGBM)-->
<!--[![Coverage Status](https://coveralls.io/repos/github/ArdalanM/pyLightGBM/badge.svg?branch=master)](https://coveralls.io/github/ArdalanM/pyLightGBM?branch=master)-->
<!--[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)]()-->

Features:
 - Classification from numpy array
 - Classification from svm format
 - Predict from existing model

Installation
------------
- Install DeepDetect (instruction [here](https://deepdetect.com/overview/installing/)):
- Install pyDD: ```pip install git+https://gitlab.classmatic.net/ardalan.mehrani/pyDD.git```

Examples
--------
Make sure DeepDetect is up and running: 
```./main/dede --port 8080```

* Classification from array:

```python
from sklearn import datasets, metrics, preprocessing, model_selection
from pydd.MLP import MLPfromArray

X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = MLPfromArray(port=8081, nclasses=10, gpu=True)
clf.fit(x_train, y_train, validation_data=[(x_test, y_test)])

print("Accuracy: ", metrics.accuracy_score(y_test, clf.predict(x_test)))
print("Log loss: ", metrics.log_loss(y_test, clf.predict_proba(x_test)))
```

- Classification from svm:

```python
from pydd.MLP import MLPfromSVM

train_path, test_path = 'x_train.svm', 'x_test.svm'
params = {'port': 8081, 'nclasses': 10, 'layers': [100, 100], 'activation': 'relu',
          'dropout': 0.5, 'db': True, 'gpu': True}

clf = MLPfromSVM(**params)
y_prob = clf.predict_proba(test_path)
y_pred = clf.predict(test_path)
```