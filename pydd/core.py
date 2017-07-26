#. -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:



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
"""

import os
import json
import time
import tempfile
import numpy as np
from pydd.utils import time_utils, os_utils
from pydd.utils.dd_client import DD
from pydd.utils.dd_utils import to_array


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

        assert json_dump['status']['code'] == 201 and json_dump['status']['msg'] == "Created", json_dump
        return json_dump

    def post_train(self, sname, data, train_parameters_input, train_parameters_mllib,
                   train_parameters_output, async):
        json_dump = self.dd.post_train(sname, data, train_parameters_input, train_parameters_mllib,
                                       train_parameters_output, async=async)

        assert json_dump['status']['code'] == 201 and json_dump['status']['msg'] == "Created", json_dump

        return json_dump

    def delete_train(self, sname, job=1):
        self.dd.delete_train(sname, job=job)
        return self

    def post_predict(self, sname, data, predict_parameters_input, predict_parameters_mllib,
                     predict_parameters_output):
        json_dump = self.dd.post_predict(sname, data, predict_parameters_input,
                                         predict_parameters_mllib, predict_parameters_output)

        assert json_dump['status']['code'] == 200 and json_dump['status']['msg'] == "OK", json_dump

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

    def get_info(self):
        json_dump = self.dd.info()
        return json_dump


class AbstractModels(AbstractDDCalls):
    def __init__(self, host="localhost", port=8080, sname="", description="", mllib="caffe",
                 service_parameters_input=None,
                 service_parameters_mllib=None,
                 service_parameters_output=None,
                 model=None,
                 tmp_dir=None):

        self.host = host
        self.port = port
        self.sname = sname
        self.model = model
        self.description = description
        self.mllib = mllib
        self.tmp_dir = tmp_dir

        self.service_parameters_input = service_parameters_input
        self.service_parameters_mllib = service_parameters_mllib
        self.service_parameters_output = service_parameters_output

        self.n_pred = 0
        self.n_fit = 0
        self.calls = []
        self.answers = []
        # self.train_logs = None
        super(AbstractModels, self).__init__(self.host, self.port)

        if self.sname:
            for service in self.get_info()['head']['services']:
                if service['name'] == self.sname.lower(): # DD lowercases services' name
                    self.delete_service(self.sname, clear="mem")
        else:
            self.sname = "pyDD_{}".format(time_utils.fulltimestamp())
            self.description = self.sname

        # Check if a repository is given otherwise creates one
        if "repository" not in self.model or not self.model["repository"]:
            self.repository = tempfile.mkdtemp(prefix="pydd_", dir=self.tmp_dir)
            self.model["repository"] = self.repository
            os_utils._create_dirs([self.model["repository"]])
        else:
            assert os.path.exists(self.model["repository"]), "{} does not exist".format(self.model["repository"])

        json_dump = self.create_service(self.sname, self.model, self.description, self.mllib,
                                        self.service_parameters_input,
                                        self.service_parameters_mllib,
                                        self.service_parameters_output)
        self.answers.append(json_dump)

        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

    def _train(self, data, parameters_input, parameters_mllib, parameters_output,
               display_metric_interval, async):

        if self.n_fit > 0:
            self.delete_service(self.sname, "mem")
            if "template" in self.service_parameters_mllib:
                self.service_parameters_mllib.pop("template")

            json_dump = self.create_service(self.sname, self.model, self.description, self.mllib,
                                            self.service_parameters_input,
                                            self.service_parameters_mllib,
                                            self.service_parameters_output)

        json_dump = self.post_train(self.sname, data,
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
                    logs = train_status["body"]["measure"]
                    print(logs, flush=True)
                    if logs:
                        train_logs.append(logs)
                else:
                    print(train_status, flush=True)
                    break

        return train_logs

    def _predict_proba(self, data, parameters_input=None, parameters_mllib=None, parameters_output=None):

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


if __name__ == "__main__":
    """ Simple unit test """
    from sklearn import datasets, preprocessing

    # Parameters
    n_classes = 10

    X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    tr_f = os.path.abspath('x_train.svm')
    datasets.dump_svmlight_file(X, y, tr_f)

    model = {"templates": "../templates/caffe", "repository": ""}
    service_parameters_input = {"connector": "svm"}
    service_parameters_output = {}
    service_parameters_mllib = {"nclasses": n_classes,
                                "gpu": False,
                                "template": "mlp", "layers": [100],
                                "activation": "relu", "dropout": 0.5, "db": True}
    clf = AbstractModels(host="localhost", port=8085, description="", mllib="caffe",
                         service_parameters_input=service_parameters_input,
                         service_parameters_mllib=service_parameters_mllib,
                         service_parameters_output=service_parameters_output,
                         model=model,
                         tmp_dir=None)

    train_parameters_input = {"db": True},
    train_parameters_output = {"measure": ["accp", "mcll"]},
    train_parameters_mllib = {
        "gpu": service_parameters_mllib["gpu"],
        "solver": {"iterations": 1000,
                   "base_lr": 0.01,
                   "solver_type": "SGD"},
        "net": {"batch_size": 128},
    }

    clf._train([tr_f], train_parameters_input, train_parameters_mllib, train_parameters_output,
               display_metric_interval=1, async=True)

    json_dump = clf._predict_proba([tr_f], parameters_output={"best": -1})

    os_utils._remove_files([tr_f])
