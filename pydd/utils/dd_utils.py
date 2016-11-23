# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pydd.utils.dd_client import DD


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


def to_array(json_dump, nclasses):
    # print(json_dump)
    nb_rows = len(json_dump['body']['predictions'])
    nb_col = nclasses

    y_score = np.zeros((nb_rows, nb_col))

    for i, row in enumerate(json_dump['body']['predictions']):
        row_number = int(row['uri'])
        assert row_number == i

        # print(row['classes'])
        for classe in row['classes']:
            class_label = int(classe['cat'])
            class_prob = classe['prob']
            y_score[row_number, class_label] = class_prob
    return y_score


