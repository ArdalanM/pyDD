pyDD: python binding for [DeepDetect](https://github.com/beniz/deepdetect)
============================================================================
<!--[![Build Status](https://travis-ci.org/ArdalanM/pyLightGBM.svg?branch=feat_ci)](https://travis-ci.org/ArdalanM/pyLightGBM)-->
<!--[![Coverage Status](https://coveralls.io/repos/github/ArdalanM/pyLightGBM/badge.svg?branch=master)](https://coveralls.io/github/ArdalanM/pyLightGBM?branch=master)-->
<!--[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)]()-->

Features:
 - Seamless integration with scikit-learn (`GridSearchCV`, `cross_val_score`, etc...)
 - Classification from array/svm (`clf.fit(X, Y)` or `clf.fit("path/to/data.svm`)
 - Predict from existing model

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
from sklearn import datasets, metrics, preprocessing, model_selection
from pydd.MLP import MLPfromArray

X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = MLPfromArray(port=8080, nclasses=10, gpu=True)
clf.fit(x_train, y_train, validation_data=[(x_test, y_test)])

print("Accuracy: ", metrics.accuracy_score(y_test, clf.predict(x_test)))
print("Log loss: ", metrics.log_loss(y_test, clf.predict_proba(x_test)))
```

- Classification from svm:

```python
from pydd.MLP import MLPfromSVM

train_path, test_path = 'x_train.svm', 'x_test.svm'
params = {'port': 8080, 'nclasses': 10, 'layers': [100, 100], 'activation': 'relu',
          'dropout': 0.5, 'db': True, 'gpu': True}

clf = MLPfromSVM(**params)
y_prob = clf.predict_proba(test_path)
y_pred = clf.predict(test_path)
```

- Grid Search:

```python
from pydd.MLP import MLPfromArray
from sklearn import datasets, model_selection, preprocessing, metrics

X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)

# Parameters
params = {'port': 8080, 'nclasses': 10,
            'layers': [100],
          'activation': 'relu', 'dropout': 0.1, 'db': True, 'gpu': True}

param_grid = {'dropout': [0.1, 0.8],'layers': [[10], [100]]}

clf = MLPfromArray(**params)

scorer = metrics.make_scorer(metrics.accuracy_score)
skf = model_selection.StratifiedKFold(n_splits=3)

grid = model_selection.GridSearchCV(clf, param_grid, scoring=scorer, cv=skf)
grid.fit(X, Y)
```