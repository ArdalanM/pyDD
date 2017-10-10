# -*- coding: utf-8 -*-
"""
@author: Evgeny Bazarov <baz.evgenii@gmail.com>
@brief:
"""

import os
import time
import json
import argparse
from pydd.utils.dd_client import DD
from pydd.utils.dd_board_logger import DDBoard
from pydd.utils import os_utils

# curl -X PUT "http://localhost:8100/services/affinitas" -d "{\"mllib\":\"caffe\",\"description\":\"classification service\"," \
#                                                           "\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"svm\"}," \
#                                                           "\"mllib\":{\"db\":true,\"template\":\"mlp\",\"nclasses\":2," \
#                                                           "\"layers\":[2048,1024,512,512,256],\"activation\":\"relu\",\"dropout\":0.2}}," \
#                                                           "\"model\":{\"templates\":\"../templates/caffe/\"," \
#                                                           "\"repository\":\"/home/ardalan.mehrani/models/\"}}"
#
# curl -X POST "http://localhost:8100/train" -d "{\"service\":\"affinitas\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{\"iterations\":7000,\"test_interval\":500,\"base_lr\":0.1,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":512}},\"input\":{\"db\":true},\"output\":{\"measure\":[\"mcll\",\"f1\",\"auc\"]}},\"data\":[\"/home/ardalan.mehrani/train.svm\",\"/home/ardalan.mehrani/test.svm\"]}"


def get_mlp_args():
    ####### Parameters
    parser = argparse.ArgumentParser("Train MLP")

    ### CONNECTION SETTINGS
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)

    ####### Service creation
    parser.add_argument("--sname", type=str, default="trainmlp")
    parser.add_argument("--description", type=str, default="classifier")
    parser.add_argument("--mllib", type=str, default="caffe")

    ### MODEL
    parser.add_argument("--repository", type=str, default="/home/ardalan.mehrani/projects/pyDD/testmlp")
    parser.add_argument("--templates", type=str, default="../templates/caffe")
    parser.add_argument("--weights", type=str, default=None)

    ### INPUT
    parser.add_argument("--connector", type=str, default="svm")

    ### MLLIB
    parser.add_argument("--nclasses", type=int, default=None)
    parser.add_argument("--ntargets", type=int, default=None)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--regression", type=bool, default=False)
    parser.add_argument("--finetuning", type=bool, default=False)
    parser.add_argument("--template", type=str, default="mlp")

    parser.add_argument("--layers", nargs='+', type=float, default=[128, 128])
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--dropout", type=float, default=0.2)
    ####### Training
    parser.add_argument("--data", nargs='+', type=str)

    ### INPUT
    parser.add_argument("--db", type=bool, default=True)

    ### MLLIB
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--class_weights", nargs='+', type=float, default=None)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_initialization", type=bool, default=False)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=70000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--solver_type", type=str, default="ADAM")
    parser.add_argument("--iter_size", type=int, default=1)

    ### OUTPUT
    parser.add_argument('--path_log', type=str, default="test")
    parser.add_argument('--measure', nargs='+', default=['accp', 'mcll'])
    parser.add_argument('--use_ddboard', type=bool, default=True)
    args = parser.parse_args()
    return args


def get_vdcnn_args():
    ########################################################
    ####### Parameters
    ########################################################
    parser = argparse.ArgumentParser("Very Deep Mixed Convolutional Neural Network (VDMCNN)")

    ### CONNECTION SETTINGS
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)

    ####### Service creation
    parser.add_argument("--sname", type=str)
    parser.add_argument("--description", type=str, default="classifier")
    parser.add_argument("--mllib", type=str, default="caffe")

    ### MODEL
    parser.add_argument("--repository", type=str)

    ### INPUT
    parser.add_argument("--connector", type=str, default="csv")
    parser.add_argument("--db", type=bool, default=False)
    parser.add_argument("--label", type=str, default="0")

    ### MLLIB
    parser.add_argument("--nclasses", type=int, default=2)

    ####### Training
    parser.add_argument("--path_data_tr", type=str)
    parser.add_argument("--path_data_te", type=str, default=None)

    ### INPUT
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--separator", type=str, default=",")
    parser.add_argument("--test_split", type=float, default=0.00)
    parser.add_argument("--label_offset", type=int, default=0)

    ### MLLIB
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--class_weights", nargs='+', type=float, default=[1.0, 1.0])
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_initialization", type=bool, default=False)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=70000)
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--solver_type", type=str, default="ADAM")
    parser.add_argument("--iter_size", type=int, default=1)

    ### OUTPUT
    parser.add_argument('--path_log', type=str)
    parser.add_argument('--measure', nargs='+')
    parser.add_argument('--use_ddboard', type=bool, default=False)

    args = parser.parse_args()
    return args


def train_mlp(opt=None):

    opt = opt if opt else get_args()

    os.makedirs(opt.repository, exist_ok=True)

    assert os.path.exists(opt.repository)
    for path in opt.data:
        assert os.path.exists(path)

    dd = DD(opt.host, opt.port)
    dd.set_return_format(dd.RETURN_PYTHON)

    ####### Service creation
    model = {"repository": opt.repository}
    if opt.weights:
        opt.model.update({"weights": opt.weights})

    parameters_input = {"connector": opt.connector}
    parameters_output = {}
    parameters_mllib = {"nclasses": opt.nclasses, "ntargets": opt.ntargets,
                        "resume": opt.resume, "layers": opt.layers,
                        "activation": opt.activation, "gpu": opt.gpu,
                        "dropout": opt.dropout, "regression": opt.regression,
                        "finetuning": opt.finetuning,
                        "weights": opt.weights,
                        "db": opt.db}

    if not opt.resume:
        model.update({"templates": opt.templates})
        parameters_mllib.update({"template": opt.template})

    if opt.gpu:
        parameters_mllib.update({"gpuid": opt.gpuid})


    dd.put_service(opt.sname,
                   model,
                   opt.description,
                   opt.mllib,
                   parameters_input,
                   parameters_mllib,
                   parameters_output)

    ####### Training
    parameters_input = {"db": opt.db, "connector": opt.connector}

    parameters_mllib = {
        "gpu": opt.gpu,
        "class_weights": opt.class_weights,
        "net": {
            "test_batch_size": opt.test_batch_size,
            "batch_size": opt.batch_size
        },
        "solver": {
            "test_initialization": opt.test_initialization,
            "base_lr": opt.base_lr,
            "iterations": opt.iterations,
            "test_interval": opt.test_interval,
            "solver_type": opt.solver_type,
            "iter_size": opt.iter_size,
            "gamma": opt.gamma
        },
        "gpuid": opt.gpuid
    }

    parameters_output = {
        "measure": opt.measure
    }

    dd.post_train(opt.sname,
                  opt.data,
                  parameters_input,
                  parameters_mllib,
                  parameters_output,
                  async=True)

    ####### Metrics check
    time.sleep(10)
    train_logs = None
    train_log = ""
    read_dd = None
    if opt.use_ddboard:
        dirname = os.path.dirname(opt.path_log)
        logdir = os.path.join(dirname, 'dd_board')
        os.makedirs(logdir, exist_ok=True)
        read_dd = DDBoard(logdir, "")

    while True:
        train_log = dd.get_train(opt.sname, job=1, timeout=1, measure_hist=True)
        if train_log["head"]["status"] == "running":
            log = train_log["body"]["measure"]
            print(log, flush=True)
            if log:
                train_logs = train_log["body"]["measure_hist"]
                if opt.use_ddboard:
                    read_dd.ddb_logger(log)
        else:
            print(train_log, flush=True)
            break

    if opt.path_log:
        print("Saving logs to", opt.path_log)
        with open(opt.path_log, "w") as f:
            json.dump(train_logs, f)


def train_vdcnn(opt=None):

    if not opt:
        opt = get_args()

    assert os.path.exists(opt.repository)
    assert os.path.exists(opt.path_data_tr)

    dd = DD(opt.host, opt.port)
    dd.set_return_format(dd.RETURN_PYTHON)

    ####### Service creation
    model = {
        "repository": opt.repository
    }
    parameters_input = {
        "connector": opt.connector,
        "db": False,
        "label": opt.label,
    }
    parameters_mllib = {
        'nclasses': opt.nclasses,
    }
    parameters_output = {}

    dd.put_service(opt.sname,
                   model,
                   opt.description,
                   opt.mllib,
                   parameters_input,
                   parameters_mllib,
                   parameters_output)

    ####### Training
    parameters_input = {
        "scale": opt.scale,
        "shuffle": opt.shuffle,
        "separator": opt.separator,
        "test_split": opt.test_split,
        "label_offset": opt.label_offset,
        "db": True
    }

    parameters_mllib = {
        "gpu": opt.gpu,
        "class_weights": opt.class_weights,
        "net": {
            "test_batch_size": opt.test_batch_size,
            "batch_size": opt.batch_size
        },
        "solver": {
            "test_initialization": opt.test_initialization,
            "base_lr": opt.base_lr,
            "iterations": opt.iterations,
            "test_interval": opt.test_interval,
            "solver_type": opt.solver_type,
            "iter_size": opt.iter_size
        },
        "gpuid": opt.gpuid
    }

    parameters_output = {"measure": opt.measure}

    data = [opt.path_data_tr]
    if opt.path_data_te:
        data.append(opt.path_data_te)

    dd.post_train(opt.sname,
                  data,
                  parameters_input,
                  parameters_mllib,
                  parameters_output,
                  async=True)

    ####### Metrics check
    time.sleep(10)
    train_logs = None
    train_log = ""
    read_dd = None
    if opt.use_ddboard:
        dirname = os.path.dirname(opt.path_log)
        logdir = os.path.join(dirname, 'dd_board')
        os.makedirs(logdir, exist_ok=True)
        read_dd = DDBoard(logdir, "")

    while True:
        train_log = dd.get_train(opt.sname, job=1, timeout=1, measure_hist=True)
        if train_log["head"]["status"] == "running":
            log = train_log["body"]["measure"]
            print(log, flush=True)
            if log:
                train_logs = train_log["body"]["measure_hist"]
                if opt.use_ddboard:
                    read_dd.ddb_logger(log)
        else:
            print(train_log, flush=True)
            break

    print("Saving logs to", opt.path_log)
    with open(opt.path_log, "w") as f:
        json.dump(train_logs, f)


def test_train_mlp():
    import os
    import numpy as np
    from sklearn import datasets, metrics, model_selection, preprocessing
    seed = 1337
    np.random.seed(seed)  # for reproducibility

    opt = get_mlp_args()

    if os.path.exists(opt.repository):
        os_utils._remove_dirs([opt.repository])

    opt.port = 8080
    opt.nclasses = 10
    opt.gpu = True
    opt.iterations = 500
    opt.solver_type = 'SGD'
    opt.base_lr = 0.01
    opt.gamma = 0.1

    # create dataset
    X, y = datasets.load_digits(n_class=opt.nclasses, return_X_y=True)
    X = preprocessing.StandardScaler().fit_transform(X)
    xtr, xte, ytr, yte = model_selection.train_test_split(X, y, test_size=0.2, random_state=seed)

    # create and save train.svm and test.svm
    tr_f = os.path.abspath('x_train.svm')
    te_f = os.path.abspath('x_test.svm')
    datasets.dump_svmlight_file(xtr, ytr, tr_f)
    datasets.dump_svmlight_file(xte, yte, te_f)

    opt.data = [tr_f, te_f]

    train_mlp(opt)

    # remove saved svm files
    os_utils._remove_files([tr_f, te_f])
    os_utils._remove_dirs([opt.repository])



if __name__ == "__main__":

    test_train_mlp()

