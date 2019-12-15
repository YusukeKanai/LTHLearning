"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import os

import matplotlib.pyplot as plt
import pandas as pd
import cupy as cp
import numpy as np
import chainer
from chainer import dataset
from chainer import datasets
from chainer import functions as F

from model_saver import load
import util

DEVICE_ID = "0"
REPORT_DIR = "experiments/accuracy_of_various_pruning_rate/result"


def get_accuracy(model, data, grand_truth_label):
    if int(DEVICE_ID) >= 0:
        data = cp.asarray(data)
        grand_truth_label = cp.asarray(grand_truth_label)
    y = model(data)
    accuracy = F.accuracy(model(data), grand_truth_label).data

    if int(DEVICE_ID) >= 0:
        accuracy = cp.asnumpy(accuracy)

    return np.asscalar(accuracy)


def main():
    device = chainer.get_device(DEVICE_ID)
    device.use()

    _, test = datasets.get_cifar10()
    test_data, test_label = dataset.concat_examples(test[:1000])

    results = []
    for pruning_rate in range(0, 100, 1):
        model = load("initial.model")
        model.to_device(device)
        masks = util.create_masks("initial.model", "final.model", pruning_rates=pruning_rate)
        model.prune_connection(masks)
        results.append([pruning_rate, get_accuracy(model, test_data, test_label)])
        do_cr = (pruning_rate + 1) % 10
        print(".", end="" if do_cr else "\n")

    df = pd.DataFrame(results, columns=['pruning_rate', 'accuracy'])
    df.to_csv(os.path.join(REPORT_DIR, 'pruning_accuracy.csv'))
    df.set_index('pruning_rate')['accuracy'].plot()
    plt.savefig(os.path.join(REPORT_DIR, 'pruning_accuracy.png'))
    plt.close('all')


if __name__ == '__main__':
    main()
