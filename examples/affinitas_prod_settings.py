# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
curl -X PUT "http://localhost:8081/services/affinitas" -d "{mllib:caffe,description:classification service,type:supervised,parameters:{input:{connector:svm},mllib:{db:true,template:mlp,nclasses:2,layers:[2048,1024,512,512,256],activation:relu,dropout:0.2}},model:{templates:../templates/caffe/,repository:/home/ardalan/Documents/ioSquare/poc/affinitas/data/dd_models/affinitas_prod}}"
curl -X POST "http://localhost:8080/train" -d "{service:affinitas,async:true,parameters:{mllib:{gpu:true,gpuid:0,solver:{iterations:7000,test_interval:500,base_lr:0.1,solver_type:ADAM},net:{batch_size:512}},input:{db:true},output:{measure:[mcll,f1,auc]}},data:[/home/ardalan/Documents/ioSquare/poc/affinitas/data/svm_data/X_train.svm,/home/ardalan/Documents/ioSquare/poc/affinitas/data/svm_data/X_test.svm]}"
curl -X GET "http://localhost:8080/services/n20"


balance cw:
{"mllib": "caffe", "parameters": {"mllib": {"activation": "relu", "dropout": 0.2, "gpu": true, "regression": false, "finetuning": false, "layers": [2048, 1024, 512, 512, 256], "template": "mlp", "nclasses": 2, "gpuid": 0, "ntargets": null, "db": true}, "output": {}, "input": {"connector": "svm"}}, "type": "supervised", "model": {"templates": "../templates/caffe", "repository": "/tmp/tmpgih87tlx/model"}, "description": "pyDD_MLP_2016-11-02-16-47-46-466356"}
{"parameters": {"mllib": {"net": {"batch_size": 512}, "solver": {"solver_type": "ADAM", "test_interval": 500, "base_lr": 0.1, "iterations": 7000}, "gpu": true, "class_weights": [1.0, 1.0]}, "output": [{"measure": ["mcll", "accp", "f1", "auc"]}], "input": [{"db": true}]}, "async": true, "data": ["/data/ardalan.mehrani/ioSquare/poc/Affinitas/data/svm_data/train_2016-11-02-16-24.svm", "/data/ardalan.mehrani/ioSquare/poc/Affinitas/data/svm_data/test_prod_2016-11-02-16-24.svm"], "service": "pyDD_MLP_2016-11-02-16-47-46-466356"}
{'auc': 0.9773534, 'mcll': 0.17840118246649725, 'train_loss': 0.12192206084728241, 'accp': 0.9388571428571428, 'precision': 0.9298999999903301, 'recall': 0.7144534116773471, 'f1': 0.8080625768131713, 'iteration': 6999.0, 'acc': 0.9388571428571428}
{'status': {'msg': 'OK', 'code': 200}, 'head': {'job': 1, 'status': 'finished', 'time': 1105.0, 'method': '/train'}, 'body': {'measure': {'auc': 0.976986, 'mcll': 0.16580418879241499, 'train_loss': 0.12192206084728241, 'accp': 0.941047619047619, 'precision': 0.9310499999903289, 'recall': 0.7192527573023181, 'f1': 0.8115605172689133, 'iteration': 6999.0, 'acc': 0.941047619047619}}}

class_weights=[1., 2.]
{"model": {"repository": "/tmp/tmpfjvnt9ne/model", "templates": "../templates/caffe"}, "parameters": {"output": {}, "input": {"connector": "svm"}, "mllib": {"ntargets": null, "db": true, "template": "mlp", "regression": false, "dropout": 0.2, "activation": "relu", "gpu": true, "finetuning": false, "nclasses": 2, "gpuid": 0, "layers": [2048, 1024, 512, 512, 256]}}, "description": "pyDD_MLP_2016-11-02-17-06-47-428196", "type": "supervised", "mllib": "caffe"}
{"data": ["/data/ardalan.mehrani/ioSquare/poc/Affinitas/data/svm_data/train_2016-11-02-16-24.svm", "/data/ardalan.mehrani/ioSquare/poc/Affinitas/data/svm_data/test_prod_2016-11-02-16-24.svm"], "parameters": {"output": [{"measure": ["mcll", "accp", "f1", "auc"]}], "input": [{"db": true}], "mllib": {"net": {"batch_size": 512}, "gpu": true, "class_weights": [1.0, 2.0], "solver": {"base_lr": 0.1, "solver_type": "ADAM", "test_interval": 500, "iterations": 7000}}}, "service": "pyDD_MLP_2016-11-02-17-06-47-428196", "async": true}
{'iteration': 6999.0, 'train_loss': 0.18101488053798676, 'precision': 0.9241499999906018, 'accp': 0.9532380952380952, 'mcll': 0.1464681947995281, 'recall': 0.7497406030317115, 'auc': 0.9776427, 'f1': 0.8278590931016462, 'acc': 0.9532380952380952}
"""

import os
from pydd.MLP import MLP

svm_folder = "/data/ardalan.mehrani/ioSquare/poc/Affinitas/data/svm_data"
model_repo = '{}/dd_cw5'.format(svm_folder)
sname = 'dd_cw5'
class_weights = [1., 5.]
data = [
    os.path.join(svm_folder, "train_2016-11-02-16-24.svm"),
    os.path.join(svm_folder, "test_prod_2016-11-02-16-24.svm")
]

clf = MLP(port=8081, nclasses=2, sname=sname, repository=model_repo,
          layers=[2048, 1024, 512, 512, 256], activation='relu',
          dropout=0.2, db=True, gpu=True)

clf.fit(data, iterations=7000, batch_size=512, base_lr=0.1, solver_type='ADAM', test_interval=500,
        metrics=['mcll', 'accp', 'f1', 'auc'], class_weights=class_weights)




