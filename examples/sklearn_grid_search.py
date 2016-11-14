# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
from sklearn import datasets, model_selection, preprocessing, metrics
from pydd.MLP import MLPfromArray

X, Y = datasets.load_digits(return_X_y=True)
X = preprocessing.StandardScaler().fit_transform(X)

# Parameters
seed = 1337
nclasses = 10
test_size = 0.2
params = {'port': 8081, 'nclasses': nclasses,
          'layers': [100], 'activation': 'relu',
          'dropout': 0.1, 'db': True, 'gpu': True}

clf = MLPfromArray(**params)


param_grid = {
    'dropout': [0.1, 0.8],
    'layers': [[10], [100]]
}

scorer = metrics.make_scorer(metrics.accuracy_score)
skf = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

grid = model_selection.GridSearchCV(clf, param_grid, scoring=scorer, cv=skf)
grid.fit(X, Y)

print("Best score: {}".format(grid.best_score_))
print("Best params: {}".format(grid.best_params_))
