# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import shutil
from scipy import sparse
from sklearn.datasets import dump_svmlight_file
from pydd.utils import time_utils
from pydd.core import AbstractMLP, AbstractXGB
from pydd.utils.dd_utils import ndarray_to_sparse_strings, sparse_to_sparse_strings


class XGB(AbstractXGB):
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
        super(XGB, self).__init__(self.host, self.port)

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

    def fit(self, train_data, validation_data=[],
            objective="multi:softprob",
            booster="gbtree",
            eval_metric="auc",
            base_score=0.5,
            seed=0,
            nthread=-1,
            iterations=10,
            test_interval=1,
            save_period=0,
            metrics=["auc", "acc"],
            async=True,
            display_metric_interval=1,
            **booster_params):

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
            "booster_params": booster_params,
        }

        self.data = []

        if train_data.name == "svm":
            self.data.extend(train_data.paths)

            if validation_data:
                for conn in validation_data:
                    self.data.extend(conn.paths)

        elif train_data.name == "array":
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(train_data.X, train_data.Y, train_f)
            self.data.append(train_f)

            if validation_data:
                for i, conn in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(conn.X, conn.Y, valid_f)
                    self.data.append(valid_f)

        self.train_logs = self._fit(self.data,
                                    self.train_parameters_input,
                                    self.train_parameters_mllib,
                                    self.train_parameters_output, display_metric_interval, async)

        return self.train_logs


class MLP(AbstractMLP):
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
        super(MLP, self).__init__(host=host,
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

    def fit(self, train_data, validation_data=[], solver=None, batch_size=128, class_weights=None,
            metrics=["mcll", "accp"], async=True, display_metric_interval=1):

        # df: True otherwise core dump when training on svm data
        self.train_parameters_input = {"db": True}
        self.train_parameters_input.update(train_data.train_parameters_input)

        self.train_parameters_output = {"measure": metrics},

        self.train_parameters_mllib = {
            "gpu": self.service_parameters_mllib["gpu"],
            "solver": solver.__dict__,
            "net": {"batch_size": batch_size},
            "class_weights": class_weights if class_weights else [1.] * self.service_parameters_mllib["nclasses"]
        }

        self.data = []

        if train_data.name == "svm":
            self.data.extend(train_data.paths)

            if validation_data:
                for conn in validation_data:
                    self.data.extend(conn.paths)

            if train_data.lmdb_paths:
                assert os.path.exists(train_data.vocab_path)
                assert len(train_data.lmdb_paths) == len(train_data.data)
                if len(train_data.lmdb_paths) == 2:
                    os.symlink(train_data.lmdb_paths[0], os.path.join(self.repository, "train.lmdb"))
                    os.symlink(train_data.lmdb_paths[1], os.path.join(self.repository, "test.lmdb"))
                else:
                    os.symlink(train_data.lmdb_paths[0], os.path.join(self.repository, "train.lmdb"))

        elif train_data.name == "array":
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(train_data.X, train_data.Y, train_f)
            self.data.append(train_f)

            if validation_data:
                for i, conn in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(conn.X, conn.Y, valid_f)
                    self.data.append(valid_f)

        self.train_logs = self._fit(self.data,
                                    self.train_parameters_input,
                                    self.train_parameters_mllib,
                                    self.train_parameters_output, display_metric_interval, async)

        if train_data.lmdb_paths:
            os.remove(os.path.join(self.repository, "vocab.dat"))
            shutil.copy(train_data.vocab_path, os.path.join(self.repository, "vocab.dat"))

        return self.train_logs

    def predict_proba(self, connector, batch_size=128):

        nclasses = self.service_parameters_mllib["nclasses"]
        self.predict_parameters_input = connector.predict_parameters_input

        self.predict_parameters_mllib = {
            "gpu": self.service_parameters_mllib["gpu"],
            "net": {"test_batch_size": batch_size}}
        if self.gpuid:
            self.predict_parameters_mllib.update({"gpuid": self.service_parameters_mllib["gpuid"]})

        self.predict_parameters_output = {"best": nclasses}

        if connector.name == "svm":
            data = connector.data

        elif connector.name == "array":
            if type(connector.X) == np.ndarray:
                data = ndarray_to_sparse_strings(connector.X)
            elif sparse.issparse(connector.X):
                data = sparse_to_sparse_strings(connector.X)

        y_score = self._predict_proba(data,
                                      connector.predict_parameters_input,
                                      self.predict_parameters_mllib,
                                      self.predict_parameters_output)

        return y_score

    def predict(self, connector, batch_size=128):
        y_score = self.predict_proba(connector, batch_size)
        return (np.argmax(y_score, 1)).reshape(len(y_score), 1)


if __name__ == "__main__":
    import numpy as np
    from pydd.Solver import GenericSolver
    from sklearn import datasets, model_selection, preprocessing
    from pydd.Connectors import SVMConnector, ArrayConnector

    # Parameters
    seed = 1337
    np.random.seed(seed)  # for reproducibility
    n_classes = 10
    params = {"port": 8085, "nclasses": n_classes, "gpu": True}
    split_params = {"test_size": 0.2, "random_state": seed}

    X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, **split_params)

    train_data = ArrayConnector(x_train, y_train)
    val_data = ArrayConnector(x_train, y_train)

    clf = MLP(**params)
    solver = GenericSolver(iterations=100, solver_type="SGD", base_lr=0.01, gamma=0.1, stepsize=30, momentum=0.9)
    clf.fit(train_data, validation_data=[val_data], solver=solver)
    clf.predict_proba(train_data)