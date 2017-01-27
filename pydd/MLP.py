# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import shutil
import json
import time
import tempfile
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.datasets import dump_svmlight_file
from pydd.utils import os_utils, time_utils
from pydd.utils.dd_utils import (AbstractDDCalls,
                                 to_array,
                                 ndarray_to_sparse_strings,
                                 sparse_to_sparse_strings)


class genericMLP(AbstractDDCalls, BaseEstimator):
    def __init__(self, host='localhost',
                 port=8080,
                 sname='',
                 mllib='caffe',
                 description='',
                 repository='',
                 templates='../templates/caffe',
                 connector='svm',
                 nclasses=None,
                 ntargets=None,
                 gpu=False,
                 gpuid=0,
                 template='mlp',
                 layers=[50],
                 activation='relu',
                 dropout=0.5,
                 regression=False,
                 finetuning=False,
                 db=True,
                 tmp_dir=None):
        self.host = host
        self.port = port
        self.sname = sname
        self.mllib = mllib
        self.description = description
        self.repository = repository
        self.templates = templates
        self.connector = connector
        self.nclasses = nclasses
        self.ntargets = ntargets
        self.gpu = gpu
        self.gpuid = gpuid
        self.template = template
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.regression = regression
        self.finetuning = finetuning
        self.db = db
        self.tmp_dir = tmp_dir

        self.params = {
            'host': self.host,
            'port': self.port,
            'sname': self.sname,
            'mllib': self.mllib,
            'description': self.description,
            'repository': self.repository,
            'templates': self.templates,
            'connector': self.connector,
            'nclasses': self.nclasses,
            'ntargets': self.ntargets,
            'gpu': self.gpu,
            'gpuid': self.gpuid,
            'template': self.template,
            'layers': self.layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'regression': self.regression,
            'finetuning': self.finetuning,
            'db': self.db,
        }
        super(genericMLP, self).__init__(self.host, self.port)

        self.n_pred = 0
        self.n_fit = 0
        self.calls = []
        self.answers = []

        if self.sname:
            self.delete_service(self.sname, "mem")
        else:
            self.sname = "pyDD_MLP_{}".format(time_utils.fulltimestamp())
            self.description = self.sname

        if not self.repository:
            self.repository = tempfile.mkdtemp(prefix="pydd_", dir=self.tmp_dir)
            os_utils._create_dirs([self.repository])

        self.model = {'templates': self.templates, 'repository': self.repository}
        self.service_parameters_mllib = {'nclasses': self.nclasses, 'ntargets': self.ntargets,
                                         'gpu': self.gpu, 'gpuid': self.gpuid,
                                         'template': self.template, 'layers': self.layers,
                                         'activation': self.activation,
                                         'dropout': self.dropout, 'regression': self.regression,
                                         'finetuning': self.finetuning, 'db': self.db}
        self.service_parameters_input = {'connector': self.connector}
        self.service_parameters_output = {}

        json_dump = self.create_service(self.sname, self.model, self.description, self.mllib,
                                        self.service_parameters_input, self.service_parameters_mllib,
                                        self.service_parameters_output)
        self.answers.append(json_dump)

        with open("{}/model.json".format(self.model['repository'])) as f:
            self.calls = [json.loads(line, encoding='utf-8') for line in f]

    def fit(self, X, Y=None, validation_data=[], lmdb_paths=[], vocab_path='', iterations=100, test_interval=None,
            solver_type='SGD',
            base_lr=0.1,
            lr_policy=None,
            stepsize=None,
            momentum=None,
            weight_decay=None,
            power=None,
            gamma=None,
            iter_size=1,
            batch_size=128,
            metrics=['mcll', 'accp'],
            class_weights=None):

        self.filepaths = []
        if type(X) == np.ndarray or sparse.issparse(X):
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(X, Y, train_f)
            self.filepaths.append(train_f)

            if len(validation_data) > 0:
                for i, (x_val, y_val) in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(x_val, y_val, valid_f)
                    self.filepaths.append(valid_f)

        elif type(X) == list:
            self.filepaths = X
            if lmdb_paths:
                assert os.path.exists(vocab_path)
                assert len(lmdb_paths) == len(X) <= 2
                if len(lmdb_paths) == 2:
                    os.symlink(lmdb_paths[0], os.path.join(self.repository, "train.lmdb"))
                    os.symlink(lmdb_paths[1], os.path.join(self.repository, "test.lmdb"))
                else:
                    os.symlink(lmdb_paths[0], os.path.join(self.repository, "train.lmdb"))
        elif type(X) == str:
            self.filepaths = [X]
        else:
            raise

        # df: True otherwise core dump when training on svm data
        self.train_parameters_input = {'db': True},
        self.train_parameters_output = {"measure": metrics},
        self.train_parameters_mllib = {
            'gpu': self.service_parameters_mllib['gpu'],
            'solver': {'iterations': iterations,
                       'test_interval': test_interval,
                       'base_lr': base_lr,
                       'solver_type': solver_type,
                       'lr_policy': lr_policy,
                       'stepsize': stepsize,
                       'momentum': momentum,
                       'weight_decay': weight_decay,
                       'power': power,
                       'gamma': gamma,
                       'iter_size': iter_size},
            'net': {'batch_size': batch_size},
            'class_weights': class_weights if class_weights else [1.] * self.service_parameters_mllib['nclasses']
        }

        if self.n_fit > 0:
            self.delete_service(self.sname, "mem")
            if 'template' in self.service_parameters_mllib:
                self.service_parameters_mllib.pop('template')

            self.create_service(self.sname, self.model, self.description, self.mllib,
                                self.service_parameters_input,
                                self.service_parameters_mllib,
                                self.service_parameters_output)

        json_dump = self.post_train(self.sname,
                                    self.filepaths,
                                    self.train_parameters_input,
                                    self.train_parameters_mllib,
                                    self.train_parameters_output, async=True)
        time.sleep(1)
        self.answers.append(json_dump)
        with open("{}/model.json".format(self.model['repository'])) as f:
            self.calls = [json.loads(line, encoding='utf-8') for line in f]

        self.n_fit += 1

        train_status = ''
        while True:
            train_status = self.get_train(self.sname, job=1, timeout=2)
            if train_status['head']['status'] == 'running':
                print(train_status['body']['measure'])
            else:
                print(train_status)
                break

        if lmdb_paths:
            os.remove(os.path.join(self.repository, "vocab.dat"))
            shutil.copy(vocab_path, os.path.join(self.repository, "vocab.dat"))

    def predict_proba(self, X, batch_size=128):

        data = [X]
        if type(X) == np.ndarray:
            data = ndarray_to_sparse_strings(X)
        elif sparse.issparse(X):
            data = sparse_to_sparse_strings(X)

        nclasses = self.service_parameters_mllib['nclasses']
        self.predict_parameters_input = {}
        self.predict_parameters_mllib = {"gpu": self.service_parameters_mllib['gpu'],
                                         "gpuid ": self.service_parameters_mllib['gpuid'],
                                         'net': {'test_batch_size': batch_size}}
        self.predict_parameters_output = {'best': nclasses}

        json_dump = self.post_predict(self.sname, data, self.predict_parameters_input,
                                      self.predict_parameters_mllib, self.predict_parameters_output)

        self.answers.append(json_dump)
        with open("{}/model.json".format(self.model['repository'])) as f:
            self.calls = [json.loads(line, encoding='utf-8') for line in f]

        y_score = to_array(json_dump, nclasses)

        return y_score

    def predict(self, X, batch_size=128):

        y_score = self.predict_proba(X, batch_size)
        return (np.argmax(y_score, 1)).reshape(len(y_score), 1)

    def get_params(self, deep=True):
        params = self.params
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self


class MLPfromSVM(genericMLP):
    def __init__(self, host='localhost',
                 port=8080,
                 sname='',
                 mllib='caffe',
                 description='',
                 repository='',
                 templates='../templates/caffe',
                 connector='svm',
                 nclasses=None,
                 ntargets=None,
                 gpu=False,
                 gpuid=0,
                 template='mlp',
                 layers=[50],
                 activation='relu',
                 dropout=0.5,
                 regression=False,
                 finetuning=False,
                 db=True,
                 tmp_dir=None):
        super(MLPfromSVM, self).__init__(host=host,
                                         port=port,
                                         sname=sname,
                                         mllib=mllib,
                                         description=description,
                                         repository=repository,
                                         templates=templates,
                                         connector=connector,
                                         nclasses=nclasses,
                                         ntargets=ntargets,
                                         gpu=gpu,
                                         gpuid=gpuid,
                                         template=template,
                                         layers=layers,
                                         activation=activation,
                                         dropout=dropout,
                                         regression=regression,
                                         finetuning=finetuning,
                                         db=db,
                                         tmp_dir=tmp_dir)


class MLPfromArray(genericMLP):
    def __init__(self, host='localhost',
                 port=8080,
                 sname='',
                 mllib='caffe',
                 description='',
                 repository='',
                 templates='../templates/caffe',
                 connector='svm',
                 nclasses=None,
                 ntargets=None,
                 gpu=False,
                 gpuid=0,
                 template='mlp',
                 layers=[50],
                 activation='relu',
                 dropout=0.5,
                 regression=False,
                 finetuning=False,
                 db=True,
                 tmp_dir=None):
        super(MLPfromArray, self).__init__(host=host,
                                           port=port,
                                           sname=sname,
                                           mllib=mllib,
                                           description=description,
                                           repository=repository,
                                           templates=templates,
                                           connector=connector,
                                           nclasses=nclasses,
                                           ntargets=ntargets,
                                           gpu=gpu,
                                           gpuid=gpuid,
                                           template=template,
                                           layers=layers,
                                           activation=activation,
                                           dropout=dropout,
                                           regression=regression,
                                           finetuning=finetuning,
                                           db=db,
                                           tmp_dir=tmp_dir)
