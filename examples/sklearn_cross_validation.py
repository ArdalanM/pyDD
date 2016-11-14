# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
from sklearn import datasets, model_selection, preprocessing
from pydd.MLP import MLPfromArray

X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)

# Parameters
seed = 1337
nclasses = 10
test_size = 0.2
params = {'port': 8081, 'nclasses': nclasses, 'layers': [100], 'activation': 'relu',
          'dropout': 0.1, 'db': True, 'gpu': True}

clf = MLPfromArray(**params)

skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
scores = model_selection.cross_val_score(clf, X, Y, scoring='accuracy', cv=skf)
print(scores)
