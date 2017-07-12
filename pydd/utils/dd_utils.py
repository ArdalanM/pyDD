# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import numpy as np
from scipy import sparse


def to_array(json_dump, nclasses, dict_uri=None):
    # print(json_dump)
    nb_rows = len(json_dump['body']['predictions'])
    nb_col = nclasses

    y_score = np.zeros((nb_rows, nb_col), dtype=np.float32)
    use_dict = False
    # If inputs are images in a folder
    if dict_uri:
        use_dict = len(dict_uri.keys()) > 0
    lmdb = False
    # Verify if input is an LMDB
    try:
        first_index = int(json_dump['body']['predictions'][0]['uri'].split('_')[0])
        if first_index == 0:
            lmdb = True
    except:
        print('INFO: No prefix index, it seems not to be an LMDB')
        pass
        
    for i, row in enumerate(json_dump['body']['predictions']):
        if lmdb:
            row_number = int(row['uri'].split('_')[0])
        elif use_dict:
            row_number = dict_uri[row['uri']]
        else:
            row_number = int(row['uri'])
            assert row_number == i # This assertion will raise error for LMDB predictions

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
        # use `%` python string formatting it's faster than `format`
        list_svm_strings[row] += "%d:%.16g " % (col, data)

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
        # use `%` python string formatting it's faster than `format`
        svm_string = list(map(lambda idx_val: "%d:%.16g" % (idx_val[0], idx_val[1]), zip(indexes, values)))
        svm_string = " ".join(svm_string)
        list_svm_strings.append(svm_string)

    return list_svm_strings
