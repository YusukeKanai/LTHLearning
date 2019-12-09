"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

from chainer import serializers
from model import VggCNN6


def save(model_file_name, net=None):
    if not net:
        net = VggCNN6()
    serializers.save_npz("./model/" + model_file_name, net)


def load(model_file_name):
    model = VggCNN6()
    serializers.load_npz("./model/" + model_file_name, model)
    return model


if __name__ == '__main__':
    save("initial_vgg_cnn6.model")
