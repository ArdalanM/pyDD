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
import glob
import os
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
    count = 0
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)  # Sleep briefly
            # If we wait more than 10 seconds say that the training is over
            count += 1
            if count >= 100:
                break
            continue
        yield line
    
    yield None


def batch_read(thefile):
    return thefile.readlines()


def main():
    # Read args
    opt = get_args()
    logs = glob.glob(os.path.join(opt.log_dir, '*.txt'))
    if len(logs) == 0:
        logs = [opt.log_dir]

    # Parameters
    prev_iter = {}
    curr_iter = {}
    dic_plot = {}
    f = {}
    for log in logs:
        vis = Visdom(env=log)

        # Create empty plot for each metrics
        dic_plots = {}
        for metric in METRICS:
            dic_plots[metric] = vis.line(np.array([0]), np.array([0]),
                            opts=dict(legend=False, xlabel='Iteration', title=metric,
                                    marginleft=30, marginright=30, marginbottom=30, margintop=30), win=metric, env=log)

        dic_plot[log] = dic_plots
        # where the monitoring begins
        curr_iter[log], prev_iter[log] = 0, 0
        f[log] = open(log)

        # Batch read
        dic_metrics = {k: [] for k in METRICS}
        iterations = []
        lines = batch_read(f[log])
        for line in lines:
            try:
                line = json.loads(line.replace('\'', '\"'))
                curr_iter[log] = line['iteration']

                # print(line)
                if curr_iter[log] > prev_iter[log]:
                    iterations.append(curr_iter[log])
                    prev_iter[log] = curr_iter[log]

                    for metric in METRICS:
                        if metric in line:
                            dic_metrics[metric].append(line[metric])
            except:
                pass

        for metric in METRICS:
            x = np.array(iterations)
            y = np.array(dic_metrics[metric])

            if len(y) > 0:
                vis.line(y, x, win=dic_plot[log] [metric], update='append', env=log)
            # END OF Batch read
    
    generator = {}
    # stream read
    print("stream mode...")
    while True:
        to_del = []  
        for log in logs:
            generator[log] = stream_read(f[log])
            line = generator[log].__next__()
            
            # If training is done of if there is an issue remove the logging from this file
            if not line:
                to_del.append(log)
                continue

            try:
                line = json.loads(line.replace('\'', '\"'))
                curr_iter[log] = line['iteration']
                if curr_iter[log] > prev_iter[log]:
                    prev_iter[log] = curr_iter[log]
                    for metric in METRICS:
                        if metric in line:
                            x = np.array([curr_iter[log]])
                            y = np.array([line[metric]])
                            vis.line(y, x, win=dic_plot[log] [metric], update='append', env=log)
            except:
                pass
        
        # Remove logs file that are not followed anymore
        for elem in to_del:
            print("Stop watching log: " + log)
            f[log].close
            logs.remove(elem)
        if not logs:
            break
    print("All training seem to be done")
        # END OF stream read


if __name__ == "__main__":
    main()
