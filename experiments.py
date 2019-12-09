"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

import chainer
import cupy as cp
from chainer import functions as F
from chainer import datasets
from chainer import dataset

from model_saver import load
import util

_, test = datasets.get_cifar10()


def show_training_result(log_file):
    plt.figure()
    with open(log_file) as f:
        logs = json.load(f)
        results = pd.DataFrame(logs)
    results[['main/accuracy', 'validation/main/accuracy']].plot()
    plt.savefig('result/accuracy.png')
    plt.close('all')


def accuracy_by_pruned_initial_model(pruning_rates=80.0, show_report=False, use_gpu='-1'):
    is_gpu = (int(use_gpu) >= 0)
    device = chainer.get_device(use_gpu)
    device.use()

    masks = util.create_masks("initial_vgg6.model", "final_vgg6.model", pruning_rates, is_gpu)
    initial_model = load("initial_vgg6.model")
    initial_model.prune_connection(masks)
    initial_model.to_device(device)

    tests = test[:100]
    data, label = dataset.concat_examples(tests)

    if is_gpu:
        data = cp.asarray(data)
        label = cp.asarray(label)
    y = initial_model(data)
    accuracy = F.accuracy(y, label).data

    if is_gpu:
        accuracy = cp.asnumpy(accuracy)
    accuracy = np.asscalar(accuracy)
    if show_report:
        print(accuracy)
    return accuracy


def accuracy_of_various_pruning_rates(use_gpu='-1'):
    plt.figure()
    results = []

    print("start")
    for i in range(10):
        for j in range(10):
            pruning_rate = 10 * i + j
            results.append(
                [pruning_rate, accuracy_by_pruned_initial_model(pruning_rates=pruning_rate, use_gpu=use_gpu)])
            print(".", end="")
        print("")

    df = pd.DataFrame(results, columns=['pruning_rate', 'accuracy'])
    df.to_csv('result/pruning_accuracy.csv')
    df.set_index('pruning_rate')['accuracy'].plot()
    plt.savefig('result/pruning_accuracy.png')
    plt.close('all')


if __name__ == '__main__':
    show_training_result('result/log')
    # accuracy_by_pruned_initial_model(pruning_rates=90, show_report=True)
    # accuracy_of_various_pruning_rates(use_gpu="0")
