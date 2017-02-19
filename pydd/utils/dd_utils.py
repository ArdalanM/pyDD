# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import numpy as np
from pydd.utils.dd_client import DD
from scipy import sparse


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


def to_array(json_dump, nclasses):
    # print(json_dump)
    nb_rows = len(json_dump['body']['predictions'])
    nb_col = nclasses

    y_score = np.zeros((nb_rows, nb_col), dtype=np.float32)

    for i, row in enumerate(json_dump['body']['predictions']):
        row_number = int(row['uri'])
        assert row_number == i

        # print(row['classes'])
        for classe in row['classes']:
            class_label = int(classe['cat'])
            class_prob = classe['prob']
            y_score[row_number, class_label] = class_prob
    return y_score


def sparse_to_sparse_strings(X):

    X = sparse.coo_matrix(X)

    list_svm_strings = [""] * X.shape[0]
    for row, col, data in zip(X.row, X.col, X.data):
        # ".16g" is the precision of `dump_svmlight_file` function. We need to respect the same precision
        list_svm_strings[row] += "{}:{:.16g} ".format(col, data)

    list_svm_strings = list(map(lambda x: x[:-1], list_svm_strings))

    return list_svm_strings


def ndarray_to_sparse_strings(X):
        list_svm_strings = []

        for i in range(X.shape[0]):
            x = X[i, :]
            indexes = x.nonzero()[0]
            values = x[indexes]

            # where the magic happen :)
            # ".16g" is the precision of `dump_svmlight_file` function. We need to respect the same precision
            svm_string = list(map(lambda idx_val: "{}:{:.16g}".format(idx_val[0], idx_val[1]), zip(indexes, values)))
            svm_string = " ".join(svm_string)
            list_svm_strings.append(svm_string)

        return list_svm_strings
