# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief:
"""
import datetime


def timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M")
    return now_str


def fulltimestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return now_str


def timestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str


def fulltimestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")
    return now_str

