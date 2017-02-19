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
from pydd.utils.dd_client import DD
from pydd.utils.dd_utils import to_array, ndarray_to_sparse_strings, sparse_to_sparse_strings


class AbstractDDCalls(object):
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port

        dd = DD(self.host, port=self.port)
        dd.set_return_format(dd.RETURN_PYTHON)
        self.dd = dd

    def create_service(self, sname, model, description, mllib, service_parameters_input,
                       service_parameters_mllib, service_parameters_output):
        json_dump = self.dd.put_service(sname, model, description, mllib, service_parameters_input,
                                        service_parameters_mllib, service_parameters_output)
        return json_dump

    def post_train(self, sname, data, train_parameters_input, train_parameters_mllib,
                   train_parameters_output, async):
        self.dd.post_train(sname, data, train_parameters_input, train_parameters_mllib,
                           train_parameters_output, async=async)

        return self

    def delete_train(self, sname, job=1):
        self.dd.delete_train(sname, job=job)
        return self

    def post_predict(self, sname, data, predict_parameters_input, predict_parameters_mllib,
                     predict_parameters_output):
        json_dump = self.dd.post_predict(sname, data, predict_parameters_input,
                                         predict_parameters_mllib, predict_parameters_output)
        return json_dump

    def delete_service(self, sname, clear=None):
        json_dump = self.dd.delete_service(sname, clear)
        return json_dump

    def get_service(self, sname):
        json_dump = self.dd.get_service(sname)
        return json_dump

    def get_train(self, sname, job=1, timeout=0, measure_hist=False):
        json_dump = self.dd.get_train(sname, job, timeout, measure_hist)
        return json_dump


class AbstractMLP(AbstractDDCalls):
    def __init__(self, host="localhost",
                 port=8080,
                 sname="",
                 mllib="caffe",
                 description="",
                 repository="",
                 templates="../templates/caffe",
                 connector="svm",
                 nclasses=None,
                 ntargets=None,
                 gpu=False,
                 gpuid=0,
                 template="mlp",
                 layers=[50],
                 activation="relu",
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
            "host": self.host,
            "port": self.port,
            "sname": self.sname,
            "mllib": self.mllib,
            "description": self.description,
            "repository": self.repository,
            "templates": self.templates,
            "connector": self.connector,
            "nclasses": self.nclasses,
            "ntargets": self.ntargets,
            "gpu": self.gpu,
            "gpuid": self.gpuid,
            "template": self.template,
            "layers": self.layers,
            "activation": self.activation,
            "dropout": self.dropout,
            "regression": self.regression,
            "finetuning": self.finetuning,
            "db": self.db,
        }
        super(AbstractMLP, self).__init__(self.host, self.port)

        self.n_pred = 0
        self.n_fit = 0
        self.calls = []
        self.answers = []
        self.train_logs = None

        if self.sname:
            self.delete_service(self.sname, "mem")
        else:
            self.sname = "pyDD_{}_{}".format(self.template, time_utils.fulltimestamp())
            self.description = self.sname

        if not self.repository:
            self.repository = tempfile.mkdtemp(prefix="pydd_", dir=self.tmp_dir)
            os_utils._create_dirs([self.repository])

        self.service_parameters_input = {"connector": self.connector}
        self.service_parameters_output = {}

        self.model = {"templates": self.templates, "repository": self.repository}
        self.service_parameters_mllib = {"nclasses": self.nclasses, "ntargets": self.ntargets,
                                         "template": self.template, "layers": self.layers,
                                         "gpu": self.gpu, "activation": self.activation,
                                         "dropout": self.dropout, "regression": self.regression,
                                         "finetuning": self.finetuning, "db": self.db}
        if self.gpuid:
            self.service_parameters_mllib.update({"gpuid": self.gpuid})

        json_dump = self.create_service(self.sname, self.model, self.description, self.mllib,
                                        self.service_parameters_input, self.service_parameters_mllib,
                                        self.service_parameters_output)
        self.answers.append(json_dump)

        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

    def _fit(self, data, parameters_input, parameters_mllib, parameters_output,
             display_metric_interval, async):

        if self.n_fit > 0:
            self.delete_service(self.sname, "mem")
            if "template" in self.service_parameters_mllib:
                self.service_parameters_mllib.pop("template")

            self.create_service(self.sname, self.model, self.description, self.mllib,
                                self.service_parameters_input,
                                self.service_parameters_mllib,
                                self.service_parameters_output)

        json_dump = self.post_train(self.sname,
                                    data,
                                    parameters_input,
                                    parameters_mllib,
                                    parameters_output, async=async)
        time.sleep(1)
        self.answers.append(json_dump)
        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

        self.n_fit += 1

        if async:
            train_logs = []
            train_status = ""
            while True:
                train_status = self.get_train(self.sname, job=1, timeout=display_metric_interval)
                if train_status["head"]["status"] == "running":
                    train_logs = train_status["body"]["measure"]
                    self.train_logs.append(train_logs)
                    print(train_logs)
                else:
                    print(train_status)
                    break

        return train_logs

    def _predict_proba(self, data, parameters_input, parameters_mllib, parameters_output):

        json_dump = self.post_predict(self.sname, data,
                                      parameters_input,
                                      parameters_mllib,
                                      parameters_output)

        self.answers.append(json_dump)
        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

        y_score = to_array(json_dump, self.service_parameters_mllib["nclasses"])

        return y_score

    def get_params(self, deep=True):
        params = self.params
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self


class AbstractXGB(AbstractDDCalls):
    pass
