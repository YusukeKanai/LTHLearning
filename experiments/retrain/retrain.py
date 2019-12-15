"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

from collections import defaultdict
from typing import NamedTuple

import os
import matplotlib.pyplot as plt
import pandas as pd

from vgg_trainer import VGGTrainerGenerator


class Accuracy(NamedTuple):
    iteration: float = 0.0
    accuracy: float = 0.0


REPORT_DIR = "experiments/retrain/result"
results = defaultdict()


def generate_trainer(pruning_rate):
    def _postprocess(entries):
        best_accuracy = results.get(pruning_rate, Accuracy()).accuracy
        new_accuracy = entries['validation/main/accuracy']
        if new_accuracy > best_accuracy:
            iteration = entries['iteration']
            results[pruning_rate] = Accuracy(iteration=iteration, accuracy=new_accuracy)

    trainer = \
        VGGTrainerGenerator(debug=False) \
            .set_description("Retrain with pruned network") \
            .load_initial_weight_net("initial.model") \
            .set_device("0") \
            .apply_early_stop(on=True) \
            .apply_mask(init_weight_net="initial.model", final_weight_net="final.model", pruning_rate=pruning_rate) \
            .log_report(filename=None, postprocess=_postprocess) \
            .generate()
    return trainer


def main(i):
    rate_range = range(0, 100, 1)

    for pruning_rate in rate_range:
        print(f'pruning rate: {pruning_rate}')
        trainer = generate_trainer(pruning_rate)
        trainer.run()

    result_list = []
    for pruning_rate in rate_range:
        res = results[pruning_rate]
        result_list.append([pruning_rate, res.iteration, res.accuracy])

    df = pd.DataFrame(result_list, columns=['pruning_rate', "iteration", 'accuracy'])
    df.to_csv(os.path.join(REPORT_DIR, f'pruning_accuracy{i}.csv'))

    df.set_index('pruning_rate')['iteration'].plot()
    plt.savefig(os.path.join(REPORT_DIR, f'pruning_iteration{i}.png'))

    plt.figure()

    df.set_index('pruning_rate')['accuracy'].plot()
    plt.savefig(os.path.join(REPORT_DIR, f'pruning_accuracy{i}.png'))

    plt.close('all')


if __name__ == '__main__':
    main(1)
