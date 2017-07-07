# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import numpy as np

class Connectors(object):

    def __init__(self, X=None, Y=None, path="", lmdb_path="", vocab_path=""):

        self.X = X
        self.Y = Y

        self.path = path
        self.lmdb_path = lmdb_path
        self.vocab_path = vocab_path

        self.service_parameters_input = {}
        self.train_parameters_input = {}
        self.predict_parameters_input = {}


class SVMConnector(Connectors):
    """
    """
    def __init__(self, path, lmdb_path="", vocab_path=""):
        self.name = "svm"

        if path:
            if not os.path.exists(path):
                print("warning: {} does not exist".format(path))

        super(SVMConnector, self).__init__(path=path,
                                           lmdb_path=lmdb_path,
                                           vocab_path=vocab_path)


class LMDBConnector(Connectors):
    """
    """
    def __init__(self, path, lmdb_path="", vocab_path=""):
        self.name = "lmdb"

        if path:
            if not os.path.exists(path):
                print("warning: {} does not exist".format(path))

        super(LMDBConnector, self).__init__(path=path,
                                            lmdb_path=lmdb_path,
                                            vocab_path=vocab_path)


class ArrayConnector(Connectors):

    def __init__(self, X, Y=None):
        self.name = "array"
        super(ArrayConnector, self).__init__(X, Y)


class CsvConnector(Connectors):

    """
    TODO: finish this connector
    Service creation

    label	        string	        no	    N/A	    Label column name
    ignore	        array of string	yes	    empty	Array of column names to ignore
    label_offset	int	            yes	    0	    Negative offset (e.g. -1) s othat labels range from 0 onward
    separator	    string	        yes	    ’,’	    Column separator character
    id	            string	        yes	    empty	Column name of the training examples identifier field, if any
    scale	        bool	        yes	    false	Whether to scale all values into [0,1]
    categoricals	array	        yes	    empty	List of categorical variables
    db	            bool	        yes	    false	whether to gather data into a database, useful for very large datasets, allows treatment in constant-size memory

    Service Training

    label	        string	no	N/A	Label column name
    ignore	        array of string	yes	empty	Array of column names to ignore
    label_offset	int	yes	0	Negative offset (e.g. -1) s othat labels range from 0 onward
    separator	    string	yes	’,’	Column separator character
    id	            string	yes	empty	Column name of the training examples identifier field, if any
    scale	        bool	yes	false	Whether to scale all values into [0,1]
    min_vals        array	yes	empty	Instead of scale, provide the scaling parameters, as returned from a training call
    max_vals	    array	yes	empty	Instead of scale, provide the scaling parameters, as returned from a training call
    categoricals	array	yes	empty	List of categorical variables
    categoricals_mapping	object	yes	empty	Categorical mappings, as returned from a training call
    db	bool	yes	false	whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
    test_split	real	yes	0	Test split part of the dataset
    shuffle	bool	yes	false	Whether to shuffle the training set (prior to splitting)
    seed	int	yes	-1	Shuffling seed for reproducible results (-1 for random seeding)


    """


class ImageConnector(Connectors):
    """
    TODO: finish this connector
    """
    def __init__(self, path, lmdb_path="", width=227, height=227, bw=False, mean=128, std=128, test_split=0.1, shuffle=False, seed=-1):
        self.name = 'image'

        if path:
            if not os.path.exists(path):
                print("warning: {} does not exist".format(path))

        super(ImageConnector, self).__init__(path=path,
                                             lmdb_path=lmdb_path,
                                            )

        self.service_parameters_input.update({'width': width,
                                              'height': height,
                                              'bw': bw,
                                              'mean': mean,
                                              'std': std})
        self.train_parameters_input.update({'width': width,
                                            'height': height,
                                            'bw': bw,
                                            'test_split': test_split,
                                            'shuffle': shuffle,
                                            'seed': seed})
        self.predict_parameters_input.update({'width': width,
                                              'height': height,
                                              'bw': bw,
                                              'mean': mean,
                                              'std': std})
