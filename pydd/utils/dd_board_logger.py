# -*- coding:utf-8 -*-
import os
import json
from shutil import rmtree
from datetime import datetime
from tensorboard_logger import configure, log_value


class DDBoard(object):
    """Version 0.4
    Converts logs to "TensorBoard compatible" data."""

    def __init__(self, base_dir="/opt/tensorboard/runs", sub_dir=None, del_dir=False):
        """
            - base_dir = string, general cache directory used by tensorboard
            - sub_dir = string, subdirectory of the current run used by tensorboard
            - del_dir = bolean, False if ommited. If set to false, the new graph is displayed after the preceding, if any. If set to true, the tensorboard cache directory will be deleted and the new graph will be the only one to appear.
        """

        self.base_dir = base_dir
        self.sub_dir = sub_dir if sub_dir else "run-{}".format(datetime.now().strftime('%y%m%d-%H%M%S'))
        self.run_dir = os.path.join(self.base_dir, self.sub_dir)

        if del_dir:
            if os.path.isdir(self.run_dir):
                rmtree(self.run_dir)  # cleaning of the tb run directory

        # ~ configure(self.run_dir, flush_secs=DDBoard.flush_time)
        configure(self.run_dir)

    def ddb_logger(self, obs):
        """obs = the Python dict (aka JSON object) to be analyzed."""
        for key in obs:
            if key != "iteration":
                log_value(key, obs[key], int(obs["iteration"]))

    def ddb_logger_file(self, json_file):
        """json_file = the json file to be analyzed"""
        # Should we check the existence of the JSon source?
        with open(json_file, 'r') as json_src:
            for line in json_src:
                json_line = json.loads(line)
                self.ddb_logger(json_line)

if __name__ == "__main__":
    vec = DDBoard("dir", "subdir")
