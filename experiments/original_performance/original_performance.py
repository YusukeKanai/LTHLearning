"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

from chainer.training import extensions

from vgg_trainer import VGGTrainerGenerator

LOG_DIR = 'experiments/original_performance/result'


def main():
    trainer = \
        VGGTrainerGenerator() \
            .set_description("Check original network performance.") \
            .set_device("0") \
            .load_initial_weight_net('initial.model') \
            .set_output('/experiments/original_performance/result') \
            .log_report() \
            .plot_report(["main/loss", "validation/main/loss"], 'plot_loss') \
            .plot_report(["main/accuracy", "validation/main/accuracy"], 'plot_accuracy') \
            .generate()

    trainer.run()


if __name__ == '__main__':
    main()
