# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
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


class XGBClassifier(AbstractDDCalls, BaseEstimator):
    """
    XGBoost

    # General parameters:
    Parameter	    Type	    Optional    Default	        Description
    objective	    string	    yes	        multi:softprob	objective function, among multi:softprob, binary:logistic, reg:linear, reg:logistic
    booster	        string	    yes	        gbtree	        which booster to use, gbtree or gblinear
    num_feature	    int	        yes	        set by xgbbost	maximum dimension of the feature
    eval_metric	    string	    yes	        obj dependant	evaluation metric internal to xgboost
    base_score	    double	    yes	        0.5	            initial prediction score, global bias
    seed	        int	        yes	        0	            random number seed
    iterations	    int	        no	        N/A	            number of boosting iterations
    test_interval	int	        yes	        1	            number of iterations between each testing pass
    save_period	int	            yes	        0	            number of iterations between model saving to disk

    # Booster parameters
    Parameter	        Type	    Optional    Default	    Description
    eta	                double	    yes	        0.3	        step size shrinkage
    gamma	            double	    yes	        0	        minimum loss reduction
    max_depth	        int	        yes	        6	        maximum depth of a tree
    min_child_weight	int	        yes	        1	        minimum sum of instance weight
    max_delta_step	    int	        yes	        0	        maximum delta step
    subsample	        double	    yes	        1.0	        subsample ratio of traning instance
    colsample	        double	    yes	        1.0	        subsample ratio of columns when contructing each tree
    lambda	            double	    yes	        1.0	        L2 regularization term on weights
    alpha	            double	    yes	        0.0	        L1 regularization term on weights
    lambda_bias	        double	    yes	        0.0	        L2 regularization for linear booster
    tree_method	        string	    yes	        auto	    tree construction algorithm, from auto, exact, approx

    # Example calls

    curl -X PUT "http://localhost:8080/services/affxgb" -d
    "{\"mllib\":\"xgboost\",\"description\":\"classification
    service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"svm\"},\"mllib\":{\"nclasses\":2}},\"model\":{\"repository\":\"/path/to/model\"}}"

    curl -X POST "http://localhost:8080/train" -d
    "{\"service\":\"testxgb\",\"async\":true,\"parameters\":{\"mllib\":{\"iterations\":100,\"test_interval\":10,\"objective\":\"binary:logistic\",\"booster_params\":{\"max_depth\":30}},\"input\":{},\"output\":{\"measure\":[\"auc\",\"mcll\",\"f1\"]}},\"data\":[\"/path/to/X_train.svm\",\"/path/to/X_test.svm\"]}"
    """
    def __init__(self, host="localhost",
                 port=8080,
                 sname="",
                 description="",
                 repository="",
                 connector="svm",
                 mllib="xgboost",
                 nclasses=2,
                 ntargets=None,
                 tmp_dir=None):
        self.host = host
        self.port = port
        self.sname = sname
        self.mllib = mllib
        self.description = description
        self.repository = repository
        self.connector = connector
        self.nclasses = nclasses
        self.ntargets = ntargets
        self.tmp_dir = tmp_dir

        self.params = {
            "host": self.host,
            "port": self.port,
            "sname": self.sname,
            "mllib": self.mllib,
            "description": self.description,
            "repository": self.repository,
            "connector": self.connector,
            "nclasses": self.nclasses,
            "ntargets": self.ntargets,
        }
        super(XGBClassifier, self).__init__(self.host, self.port)

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

        self.model = {"repository": self.repository}
        self.service_parameters_mllib = {"nclasses": self.nclasses, "ntargets": self.ntargets}
        self.service_parameters_input = {"connector": self.connector}
        self.service_parameters_output = {}

        json_dump = self.create_service(self.sname, self.model, self.description, self.mllib,
                                        self.service_parameters_input, self.service_parameters_mllib,
                                        self.service_parameters_output)
        self.answers.append(json_dump)

        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

    def fit(self, X, Y=None, validation_data=[],
            objective="multi:softprob",
            booster="gbtree",
            eval_metric="auc",
            base_score=0.5,
            seed=0,
            nthread=-1,
            iterations=10,
            test_interval=1,
            save_period=0,
            eta=0.3, gamma=0., max_depth=6, min_child_weight=1, max_delta_step=0,
            subsample=1., colsample=1., lambda_reg=1., alpha_reg=0.,
            lambda_bias=0.0, tree_method="auto",
            metrics=["auc", "acc"]):

        self.booster_params = {"eta": eta, "gamma": gamma, "max_depth": max_depth,
                               "min_child_weight": min_child_weight, "max_delta_step": max_delta_step,
                               "subsample": subsample, "colsample": colsample,
                               "lambda": lambda_reg, "alpha": alpha_reg,
                               "lambda_bias": lambda_bias, "tree_method": tree_method}

        # df: True otherwise core dump when training on svm data
        self.train_parameters_input = {},
        self.train_parameters_output = {"measure": metrics},
        self.train_parameters_mllib = {
            "objective": objective,
            "booster": booster,
            "nthread": nthread,
            "eval_metric": eval_metric,
            "base_score": base_score,
            "seed": seed,
            "iterations": iterations,
            "test_interval": test_interval,
            "save_period": save_period,
            "booster_params": self.booster_params,
        }

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
        elif type(X) == str:
            self.filepaths = [X]
        else:
            raise

        if self.n_fit > 0:
            self.delete_service(self.sname, "mem")
            if "template" in self.service_parameters_mllib:
                self.service_parameters_mllib.pop("template")

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
        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

        self.n_fit += 1

        train_status = ""
        while True:
            train_status = self.get_train(self.sname, job=1, timeout=2)
            if train_status["head"]["status"] == "running":
                print(train_status["body"]["measure"])
            else:
                print(train_status)
                break

    def predict_proba(self, X):

        data = [X]
        if type(X) == np.ndarray:
            data = ndarray_to_sparse_strings(X)
        elif sparse.issparse(X):
            data = sparse_to_sparse_strings(X)

        nclasses = self.service_parameters_mllib["nclasses"]
        self.predict_parameters_input = {}
        self.predict_parameters_mllib = {}
        self.predict_parameters_output = {"best": nclasses}

        json_dump = self.post_predict(self.sname, data, self.predict_parameters_input,
                                      self.predict_parameters_mllib, self.predict_parameters_output)

        self.answers.append(json_dump)
        with open("{}/model.json".format(self.model["repository"])) as f:
            self.calls = [json.loads(line, encoding="utf-8") for line in f]

        y_score = to_array(json_dump, nclasses)

        return y_score

    def predict(self, X):

        y_score = self.predict_proba(X)
        return (np.argmax(y_score, 1)).reshape(len(y_score), 1)

    def get_params(self, deep=True):
        params = self.params
        return params

    def set_params(self, **kwargs):
        params = self.get_params()
        params.update(kwargs)
        self.__init__(**params)
        return self


if __name__ == "__main__":

    # Parameters
    seed = 1337
    np.random.seed(seed)  # for reproducibility
    n_classes = 10
    params = {"port": 8085, "nclasses": n_classes}
    split_params = {"test_size": 0.2, "random_state": seed}

    clf = XGBClassifier(**params)

    svm_f = [
        os.path.abspath("x_train.svm"),
        os.path.abspath("x_test.svm")
    ]

    clf.fit(svm_f, nthread=10, iterations=1000, metrics=["mcll", "cmfull"])

    y = clf.predict_proba(svm_f[0])

