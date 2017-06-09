# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
import argparse
import json

from visdom import Visdom


# Parameters
METRICS = [
    'mcll',
    'acc',
    'accp',
    'train_loss',
    'f1',
    'precision',
    'recall',
    'mcc',
    'smoothed_loss',
]

def get_args():
    parser = argparse.ArgumentParser('data viz')
    parser.add_argument("--log_dir", type=str, default='/mnt/storagebox/sharedData/poc-kaidee/logs.json', help="folder where logs are located")
    args = parser.parse_args()
    return args

def follow(thefile):
    """Iterator reading a file (line by line) and wait for new lines"""
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line

def main():
    opt = get_args()

    vis = Visdom()

    # Create empty plot for each metrics
    dic_plots = {}
    for metric in METRICS:
        dic_plots[metric] =  vis.line(np.array([0]), np.array([0]),
                                      opts=dict(legend=False, xlabel='Iteration', title=metric,
                                                marginleft=30, marginright=30, marginbottom=30, margintop=30))

    # where the monitoring begins
    curr_iter, prev_iter = 0, 0
    with open(opt.log_dir) as f:
        generator = follow(f)
        while True:
            line = generator.__next__()

            try:
                line = json.loads(line.replace('\'', '\"'))
                curr_iter = line['iteration']
                print(line)
            except:
                pass

            if curr_iter > prev_iter:

                # Fill the plots as new values are written to log files
                for metric in METRICS:
                    if metric in line:
                        value = line[metric]
                        vis.line(np.array([value]), np.array([curr_iter]), win=dic_plots[metric], update='append')

            prev_iter = curr_iter


if __name__ == "__main__":
    main()