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


# curl -X PUT "http://localhost:8100/services/affinitas" -d "{\"mllib\":\"caffe\",\"description\":\"classification service\"," \
#                                                           "\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"svm\"}," \
#                                                           "\"mllib\":{\"db\":true,\"template\":\"mlp\",\"nclasses\":2," \
#                                                           "\"layers\":[2048,1024,512,512,256],\"activation\":\"relu\",\"dropout\":0.2}}," \
#                                                           "\"model\":{\"templates\":\"../templates/caffe/\"," \
#                                                           "\"repository\":\"/home/ardalan.mehrani/models/\"}}"
#
# curl -X POST "http://localhost:8100/train" -d "{\"service\":\"affinitas\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{\"iterations\":7000,\"test_interval\":500,\"base_lr\":0.1,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":512}},\"input\":{\"db\":true},\"output\":{\"measure\":[\"mcll\",\"f1\",\"auc\"]}},\"data\":[\"/home/ardalan.mehrani/train.svm\",\"/home/ardalan.mehrani/test.svm\"]}"


def get_args():
    ####### Parameters
    parser = argparse.ArgumentParser("Train MLP")

    ### CONNECTION SETTINGS
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)

    ####### Service creation
    parser.add_argument("--sname", type=str, default='trainmlp')
    parser.add_argument("--description", type=str, default="classifier")
    parser.add_argument("--mllib", type=str, default="caffe")

    ### MODEL
    parser.add_argument("--repository", type=str, default="/home/ardalan.mehrani/projects/nlp-benchmarks/testmlp")
    parser.add_argument("--templates", type=str, default="../templates/caffe")

    ### INPUT
    parser.add_argument("--connector", type=str, default="svm")

    ### MLLIB
    parser.add_argument("--nclasses", type=int, default=2)
    parser.add_argument("--layers", nargs='+', type=float, default=[128, 128])
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--dropout", type=float, default=0.2)
    # parser.add_argument("--db", type=bool, default=True)

    ####### Training
    parser.add_argument("--path_data_tr", type=str,
                        default="/mnt/terabox/research/nlp-benchmarks/datasets/affinitas/affinitas-en-0/txtfeat@10k-catfeat@100-ngram@12-idf@true-lower@true/train.svm")
    parser.add_argument("--path_data_te", type=str,
                        default="/mnt/terabox/research/nlp-benchmarks/datasets/affinitas/affinitas-en-0/txtfeat@10k-catfeat@100-ngram@12-idf@true-lower@true/test.svm")

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


def main(opt=None):

    if not opt:
        opt = get_args()

    os.makedirs(opt.repository, exist_ok=True)

    assert os.path.exists(opt.repository)
    assert os.path.exists(opt.path_data_tr)

    dd = DD(opt.host, opt.port)
    dd.set_return_format(dd.RETURN_PYTHON)

    ####### Service creation
    model = {"repository": opt.repository, "templates": opt.templates}
    parameters_input = {"connector": opt.connector}

    parameters_mllib = {
        "nclasses": opt.nclasses,
        "layers": opt.layers,
        "activation": opt.activation,
        "gpu": opt.gpu,
        "dropout": opt.dropout,
        "db": opt.db,
        "template": 'mlp'
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
            "iter_size": opt.iter_size
        },
        "gpuid": opt.gpuid
    }

    parameters_output = {
        "measure": opt.measure
    }

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

    if opt.path_log:
        print("Saving logs to", opt.path_log)
        with open(opt.path_log, "w") as f:
            json.dump(train_logs, f)


if __name__ == "__main__":
    opt = get_args()

    main()