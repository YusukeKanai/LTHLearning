"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import chainer
import chainer.links as L
import chainer.functions as F

import cupy as cp


class Block(chainer.Chain):
    def __init__(self, input_channel, output_channel, kernel=3, stride=1, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(
                input_channel, output_channel,
                kernel, stride, pad,
                nobias=True
            )
            self.conv1 = L.Convolution2D(
                output_channel, output_channel,
                kernel, stride, pad,
                nobias=True
            )
            self.bn = L.BatchNormalization(output_channel)
        self.__weights = {
            "c1": self.conv0.W,
            "c2": self.conv1.W,
        }

    def forward(self, x):
        h = F.relu(self.conv0(x))
        h = F.relu(self.bn(self.conv1(h)))
        return h

    @property
    def weights(self):
        return self.__weights


class VggCNN6(chainer.Chain):
    def __init__(self):
        super(VggCNN6, self).__init__()
        with self.init_scope():
            self.block0 = Block(3, 64)
            self.block1 = Block(64, 128)
            self.block2 = Block(128, 256)
            self.fc1 = L.Linear(256 * 8 * 8, 256)
            self.fc2 = L.Linear(256, 10)
        self.__weights = {
            "b0": self.block0.weights,
            "b1": self.block1.weights,
            "b2": self.block1.weights,
            "fc1": self.fc1.W,
            "fc2": self.fc2.W,
        }

    def forward(self, x):
        h = self.block0(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.block1(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.block2(h)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    def prune_connection(self, masks):
        def apply_mask(layer, sub_masks):
            for c_target_name, c_layer in layer.items():
                if isinstance(c_layer, dict):
                    apply_mask(c_layer, sub_masks[c_target_name])
                    continue
                target_mask = sub_masks[c_target_name]
                # assert target_mask.shape == c_target.shape
                # assert type(c_target.data) == type(target_mask)
                xp = c_layer.xp
                if xp == cp:
                    target_mask = cp.asarray(target_mask)
                c_layer.data = xp.multiply(c_layer.data, target_mask)

        apply_mask(self.__weights, masks)

    @property
    def weights(self):
        return self.__weights
