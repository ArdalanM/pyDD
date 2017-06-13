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


def stream_read(thefile):
    """Iterator reading a file (line by line) and wait for new lines"""
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line


def batch_read(thefile):
    return thefile.readlines()


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

        # Batch read
        dic_metrics = {k:[] for k in METRICS}
        iterations = []
        lines = batch_read(f)
        for line in lines:
            try:
                line = json.loads(line.replace('\'', '\"'))
                curr_iter = line['iteration']

                # print(line)
                if curr_iter > prev_iter:
                    iterations.append(curr_iter)
                    prev_iter = curr_iter

                    for metric in METRICS:
                        if metric in line:
                            dic_metrics[metric].append(line[metric])
            except:
                pass

        for metric in METRICS:
            x = np.array(iterations)
            y = np.array(dic_metrics[metric])

            if len(y) > 0:
                vis.line(y, x, win=dic_plots[metric], update='append')
        # END OF Batch read

        # stream read
        print("stream mode...")
        generator = stream_read(f)
        while True:
            line = generator.__next__()

            try:
                line = json.loads(line.replace('\'', '\"'))
                curr_iter = line['iteration']
                if curr_iter > prev_iter:
                    prev_iter = curr_iter
                    for metric in METRICS:
                        if metric in line:
                            x = np.array([curr_iter])
                            y = np.array([line[metric]])
                            vis.line(y, x, win=dic_plots[metric], update='append')
            except:
                pass
        # END OF stream read

if __name__ == "__main__":
    main()