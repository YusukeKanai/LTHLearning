"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import numpy as np
import cupy as cp
from model_saver import load


def create_masks(initial_model, final_model, pruning_rates, is_gpu=False):
    i_layers = load(initial_model).weights
    f_layers = load(final_model).weights

    if isinstance(pruning_rates, int) or isinstance(pruning_rates, float):
        pruning_rates = [pruning_rates] * __get_layer_count(i_layers)

    def calculate_mask(input_layer, final_layer, pruning_rate):
        layer_size = input_layer.size
        prune_num = int(layer_size * pruning_rate / 100)
        # filter by large and same sign
        sign_i_weight = np.sign(input_layer.data.flatten())
        f_weight = final_layer.data.flatten()
        sort_seed = np.multiply(sign_i_weight, f_weight)

        # create filter
        xp = cp if is_gpu else np
        layer_mask = xp.ones(layer_size, dtype='float32')
        mask_target = sort_seed.argsort()[:prune_num]
        layer_mask[mask_target] = 0
        layer_mask = layer_mask.reshape(input_layer.shape)
        return layer_mask

    def create_c_masks(c_i_layers, c_f_layers, c_pruning_rates):
        offset = 0
        c_masks = dict()
        for layer_name, pruning_rate in zip(c_i_layers.keys(), c_pruning_rates):
            layer = c_i_layers[layer_name]
            if isinstance(layer, dict):
                sub_layer_count = __get_layer_count(layer)
                c_masks[layer_name] = create_c_masks(layer, c_f_layers[layer_name],
                                                     c_pruning_rates[offset:offset + sub_layer_count])
                offset += sub_layer_count
            else:
                c_masks[layer_name] = calculate_mask(layer, c_f_layers[layer_name], pruning_rate)
                offset += 1
        return c_masks

    return create_c_masks(i_layers, f_layers, pruning_rates)


def __get_layer_count(layers):
    def count(c_layers):
        count_ = 0
        for layer in c_layers.values():
            if isinstance(layer, dict):
                count_ += count(layer)
            else:
                count_ += 1
        return count_

    return count(layers)


if __name__ == '__main__':
    i_layers = load("initial_vgg6.model").weights
    assert 8 == __get_layer_count(i_layers)
