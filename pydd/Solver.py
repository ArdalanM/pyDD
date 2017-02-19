# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
import os
import json
import time
import shutil
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


class GenericSolver(object):
    """
    iterations	        int	    yes	    N/A	Max number of solver’s iterations
    snapshot	        int	    yes	    N/A	Iterations between model snapshots
    snapshot_prefix	    string	yes	    empty	Prefix to snapshot file, supports repository
    solver_type	        string	yes	    SGD	from "SGD”, “ADAGRAD”, “NESTEROV”, “RMSPROP”, “ADADELTA” and “ADAM”
    test_interval	    int	    yes	    N/A	Number of iterations between testing phases
    test_initialization	bool	true	N/A	Whether to start training by testing the network
    lr_policy	        string	yes	    N/A	learning rate policy (“step”, “inv”, “fixed”, …)
    base_lr	            real	yes	    N/A	Initial learning rate
    gamma	            real	yes	    N/A	Learning rate drop factor
    stepsize	        int	    yes	    N/A	Number of iterations between the dropping of the learning rate
    momentum	        real	yes	    N/A	Learning momentum
    weight_decay	    real	yes	    N/A	Weight decay
    power	            real	yes	    N/A	    Power applicable to some learning rate policies
    iter_size	        int	    yes	    1	Number of passes (iter_size * batch_size) at every iteration
    """
    def __init__(self, iterations=None, snapshot=None, snapshot_prefix=None, solver_type=None,
                 test_interval=None, test_initialization=True, lr_policy=None, base_lr=None, gamma=None,
                 stepsize=None, momentum=None, weight_decay=None, power=None, iter_size=1):

        self.iterations = iterations
        self.snapshot = snapshot
        self.snapshot_prefix = snapshot_prefix
        self.solver_type = solver_type
        self.test_interval = test_interval
        self.test_initialization = test_initialization
        self.lr_policy = lr_policy
        self.base_lr = base_lr
        self.gamma = gamma
        self.stepsize = stepsize
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.power = power
        self.iter_size = iter_size



















