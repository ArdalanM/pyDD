# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import shutil
import numpy as np
import glob
from scipy import sparse
from sklearn.datasets import dump_svmlight_file
from pydd.utils import time_utils
from pydd.core import AbstractModels
from pydd.utils.dd_utils import ndarray_to_sparse_strings, sparse_to_sparse_strings


class MLP(AbstractModels):
    """

    """

    def get_connector_parameters(self, connector):
        # If the connector is not an image
        if isinstance(self.connector, str):
            parameters_input = {"connector": self.connector}
        # If the connector is an image
        else:
            # Differiate paraameters if Caffe or Tensorflow is used
            bw = connector.service_parameters_input["bw"]
            mean = connector.service_parameters_input["mean"]
            std = connector.service_parameters_input["std"]
            parameters_input = {"connector": connector.name,
                "width": connector.service_parameters_input["width"],
                "height": connector.service_parameters_input["height"]
            }

            if self.mllib == 'caffe':
                parameters_input.update({"bw": bw})
                if isinstance(mean, list):
                    parameters_input.update({"mean": mean})
                        
            elif self.mllib == "tensorflow":
                parameters_input.update({"std": std})
                if isinstance(mean, float):
                    parameters_input.update({"mean": mean})
        
        return parameters_input

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
                 resume=False,
                 template="mlp",
                 layers=[100],
                 activation="relu",
                 dropout=0.5,
                 regression=False,
                 finetuning=False,
                 weights=False,
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
        self.resume = resume
        self.layers = layers
        self.activation = activation
        self.dropout = dropout
        self.regression = regression
        self.finetuning = finetuning
        self.weights = weights
        self.db = db
        self.tmp_dir = tmp_dir

        self.model = {"repository": self.repository}
        if self.weights:
            self.model.update({"weights": self.weights})

        self.service_parameters_output = {}
        self.service_parameters_mllib = {"nclasses": self.nclasses, "ntargets": self.ntargets,
                                        "resume": self.resume, "layers": self.layers,
                                        "activation": self.activation, "gpu": self.gpu,
                                        "dropout": self.dropout, "regression": self.regression,
                                        "finetuning": self.finetuning,
                                        "weights": self.weights,
                                        "db": self.db}

        self.service_parameters_input = self.get_connector_parameters(connector)

        if not self.resume:
            self.model.update({"templates": self.templates})
            self.service_parameters_mllib.update({"template": self.template})

        if self.gpu:
            self.service_parameters_mllib.update({"gpuid": self.gpuid})

        super(MLP, self).__init__(host=self.host, port=self.port, sname=self.sname,
                                  description=self.description, mllib=self.mllib,
                                  service_parameters_input=self.service_parameters_input,
                                  service_parameters_output=self.service_parameters_output,
                                  service_parameters_mllib=self.service_parameters_mllib,
                                  model=self.model, tmp_dir=self.tmp_dir)

    def fit(self, train_data, validation_data=[], solver=None, batch_size=128, class_weights=None,
            metrics=["mcll", "accp"], async=True, display_metric_interval=1):

        # df: True otherwise core dump when training on svm data
        self.train_parameters_input = {"db": True}
        self.train_parameters_input.update(train_data.train_parameters_input)
        # Remove black and white parameteer if the mllib used is tensorflow
        if self.mllib == "tensorflow" and data.name == "image":
            del self.train_parameters_input['bw']

        self.train_parameters_output = {"measure": metrics}

        self.train_parameters_mllib = {"solver": solver.__dict__,
                                       "gpu": self.gpu,
                                       "net": {"batch_size": batch_size},
                                       #"class_weights": class_weights if class_weights else [1.] * self.service_parameters_mllib["nclasses"],
                                       "resume": self.service_parameters_mllib["resume"]
                                       }
        if self.gpu:
            self.train_parameters_mllib.update({"gpuid": self.gpuid})

        self.data = []
        if train_data.name in ['svm', 'image']:
            self.data.append(train_data.path)

            if train_data.lmdb_path:
                    os.symlink(train_data.lmdb_path, os.path.join(self.repository, "train.lmdb"))

            if validation_data:

                if len(validation_data) == 1:
                    test_data = validation_data[0]
                    self.data.append(test_data.path)
                    if test_data.lmdb_path:
                        os.symlink(test_data.lmdb_path, os.path.join(self.repository, "test.lmdb"))
                else:
                    for i, connector in enumerate(validation_data):
                        self.data.append(connector.path)
                        if connector.lmdb_path:
                            os.symlink(connector.lmdb_path, os.path.join(self.repository, "test_{}.lmdb".format(i+1)))

        elif train_data.name == "array":
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(train_data.X, train_data.Y, train_f)
            self.data.append(train_f)

            if validation_data:
                for i, conn in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(conn.X, conn.Y, valid_f)
                    self.data.append(valid_f)

        self.train_logs = self._train(self.data,
                                      self.train_parameters_input,
                                      self.train_parameters_mllib,
                                      self.train_parameters_output, display_metric_interval, async)

        if train_data.lmdb_path:
            os.remove(os.path.join(self.repository, "vocab.dat"))
            shutil.copy(train_data.vocab_path, os.path.join(self.repository, "vocab.dat"))

        return self.train_logs

    def predict_proba(self, connector, batch_size=128):

        nclasses = self.service_parameters_mllib["nclasses"]
        self.predict_parameters_input = self.get_connector_parameters(connector)

        self.predict_parameters_mllib = {
            "gpu": self.service_parameters_mllib["gpu"],
            "net": {"test_batch_size": batch_size}}
        if self.gpu:
            self.predict_parameters_mllib.update({"gpuid": self.service_parameters_mllib["gpuid"]})

        self.predict_parameters_output = {"best": nclasses}

        if connector.name == "svm":
            data = [connector.path]

        if connector.name == "lmdb":
            data = [connector.path]

        if connector.name == "image":
            if os.path.isdir(connector.path):
                data = glob.glob(os.path.join(connector.path, '*'))
            else:
                data = [connector.path]

        elif connector.name == "array":
            if type(connector.X) == np.ndarray:
                data = ndarray_to_sparse_strings(connector.X)
            elif sparse.issparse(connector.X):
                data = sparse_to_sparse_strings(connector.X)

        y_score = self._predict_proba(data,
                                      self.predict_parameters_input,
                                      self.predict_parameters_mllib,
                                      self.predict_parameters_output)

        return y_score

    def predict(self, connector, batch_size=128, dict_uri=None):
        y_score = self.predict_proba(connector, batch_size, dict_uri)
        return (np.argmax(y_score, 1)).reshape(len(y_score), 1)


class LR(AbstractModels):
    """

    """

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
                 resume=False,
                 gpuid=0,
                 template="lregression",
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
        self.resume = resume
        self.regression = regression
        self.finetuning = finetuning
        self.db = db
        self.tmp_dir = tmp_dir

        self.service_parameters_output = {}
        self.model = {"repository": self.repository}
        self.service_parameters_mllib = {"nclasses": self.nclasses, "ntargets": self.ntargets, "gpu": self.gpu,
                                         "regression": self.regression, "finetuning": self.finetuning, "db": self.db}

        if not self.resume:
            self.model.update({"templates": self.templates})
            self.service_parameters_mllib.update({"template": self.template})

        self.service_parameters_input = {"connector": self.connector if isinstance(self.connector, str) else connector.name}
        if self.gpu:
            self.service_parameters_mllib.update({"gpuid": self.gpuid})

        super(LR, self).__init__(host=self.host, port=self.port, sname=self.sname,
                                 description=self.description, mllib=self.mllib,
                                 service_parameters_input=self.service_parameters_input,
                                 service_parameters_output=self.service_parameters_output,
                                 service_parameters_mllib=self.service_parameters_mllib,
                                 model=self.model, tmp_dir=self.tmp_dir)

    def fit(self, train_data, validation_data=[], solver=None, batch_size=128, class_weights=None,
            metrics=["mcll", "accp"], async=True, display_metric_interval=1):

        # df: True otherwise core dump when training on svm data
        self.train_parameters_input = {"db": True}
        self.train_parameters_input.update(train_data.train_parameters_input)

        self.train_parameters_output = {"measure": metrics}

        self.train_parameters_mllib = {"solver": solver.__dict__,
                                       "net": {"batch_size": batch_size},
                                       #"class_weights": class_weights if class_weights else [1.] * self.service_parameters_mllib["nclasses"]
                                       }
        if self.gpu:
            self.train_parameters_mllib.update({"gpu": self.gpu, "gpuid": self.gpuid})
        self.data = []

        if train_data.name == "svm":
            self.data.append(train_data.path)

            if train_data.lmdb_path:
                os.symlink(train_data.lmdb_path, os.path.join(self.repository, "train.lmdb"))

            if validation_data:
                test_data = validation_data[0]
                self.data.append(test_data.path)
                if test_data.lmdb_path:
                    os.symlink(test_data.lmdb_path, os.path.join(self.repository, "test.lmdb"))

                if len(validation_data) > 1:
                    for i, connector in enumerate(validation_data):
                        self.data.append(connector.path)
                        if connector.lmdb_path:
                            os.symlink(connector.lmdb_path, os.path.join(self.repository, "test_{}.lmdb".format(i+1)))

        elif train_data.name == "array":
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(train_data.X, train_data.Y, train_f)
            self.data.append(train_f)

            if validation_data:
                for i, conn in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(conn.X, conn.Y, valid_f)
                    self.data.append(valid_f)

        self.train_logs = self._train(self.data,
                                      self.train_parameters_input,
                                      self.train_parameters_mllib,
                                      self.train_parameters_output, display_metric_interval, async)

        if train_data.lmdb_path:
            os.remove(os.path.join(self.repository, "vocab.dat"))
            shutil.copy(train_data.vocab_path, os.path.join(self.repository, "vocab.dat"))

        return self.train_logs

    def predict_proba(self, connector, batch_size=128, dict_uri=None):

        nclasses = self.service_parameters_mllib["nclasses"]
        self.predict_parameters_input = connector.predict_parameters_input

        self.predict_parameters_mllib = {
            "gpu": self.service_parameters_mllib["gpu"],
            "net": {"test_batch_size": batch_size}}
        if self.gpu:
            self.predict_parameters_mllib.update({"gpuid": self.gpuid})

        self.predict_parameters_output = {"best": nclasses}

        if connector.name == "svm":
            data = [connector.path]

        if connector.name == "lmdb":
            data = [connector.path]

        elif connector.name == "array":
            if type(connector.X) == np.ndarray:
                data = ndarray_to_sparse_strings(connector.X)
            elif sparse.issparse(connector.X):
                data = sparse_to_sparse_strings(connector.X)

        y_score = self._predict_proba(data,
                                      connector.predict_parameters_input,
                                      self.predict_parameters_mllib,
                                      self.predict_parameters_output, dict_uri)

        return y_score

    def predict(self, connector, batch_size=128):
        y_score = self.predict_proba(connector, batch_size)
        return (np.argmax(y_score, 1)).reshape(len(y_score), 1)


class XGB(AbstractModels):
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

        self.model = {"repository": self.repository}
        self.service_parameters_mllib = {"nclasses": self.nclasses, "ntargets": self.ntargets}
        self.service_parameters_input = {"connector": self.connector if isinstance(self.connector, str) else connector.name}
        self.service_parameters_output = {}

        super(XGB, self).__init__(host=self.host, port=self.port, sname=self.sname,
                                  description=self.description, mllib=self.mllib,
                                  service_parameters_input=self.service_parameters_input,
                                  service_parameters_output=self.service_parameters_output,
                                  service_parameters_mllib=self.service_parameters_mllib,
                                  model=self.model, tmp_dir=self.tmp_dir)

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
            self.data.append(train_data.path)

            if validation_data:
                for connector in validation_data:
                    self.data.append(connector.path)

        elif train_data.name == "array":
            train_f = os.path.join(self.repository, "x_train_{}.svm".format(time_utils.fulltimestamp()))
            dump_svmlight_file(train_data.X, train_data.Y, train_f)
            self.data.append(train_f)

            if validation_data:
                for i, connector in enumerate(validation_data):
                    valid_f = os.path.join(self.repository, "x_val{}_{}.svm".format(i, time_utils.fulltimestamp()))
                    dump_svmlight_file(connector.X, connector.Y, valid_f)
                    self.data.append(valid_f)

        self.train_logs = self._train(self.data,
                                      self.train_parameters_input,
                                      self.train_parameters_mllib,
                                      self.train_parameters_output, display_metric_interval, async)
        return self.train_logs


if __name__ == "__main__":
    """Simple unit test"""
    from sklearn import datasets, model_selection, preprocessing
    from pydd.solver import GenericSolver
    from pydd.connectors import ArrayConnector, SVMConnector
    from pydd.utils import os_utils

    # Parameters
    seed = 1337
    np.random.seed(seed)  # for reproducibility
    n_classes = 10
    test_size = 0.2
    host = 'localhost'
    port = 8080
    iteration=100
    lr=0.01
    gpu=False

    X, y = datasets.load_digits(n_class=n_classes, return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
    tr_f = os.path.abspath('x_train.svm')
    te_f = os.path.abspath('x_test.svm')
    datasets.dump_svmlight_file(x_train, y_train, tr_f)
    datasets.dump_svmlight_file(x_test, y_test, te_f)

    # train_data = ArrayConnector(x_train, y_train)
    # val_data = ArrayConnector(x_train, y_train)
    train_data = SVMConnector(tr_f)
    val_data = SVMConnector(te_f)

    clf = MLP(host=host, port=port, nclasses=n_classes, layers=[100], gpu=gpu)
    solver = GenericSolver(iterations=iteration, test_interval=30, solver_type="SGD", base_lr=lr)
    clf.fit(train_data, validation_data=[val_data],  solver=solver)
    clf.predict_proba(train_data)

    clf.fit(train_data, validation_data=[val_data], solver=solver)
    y_pred = clf.predict_proba(train_data)

    clf = LR(host=host, port=port, nclasses=n_classes, gpu=gpu)
    solver = GenericSolver(iterations=iteration, solver_type="SGD", base_lr=lr)
    clf.fit(train_data, solver=solver)
    y_pred = clf.predict_proba(train_data)

    clf = XGB(host=host, port=port, nclasses=n_classes)
    # logs = clf.fit(train_data, validation_data=[val_data])

    os_utils._remove_files([tr_f, te_f])

