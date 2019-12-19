"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

from vgg_trainer import VGGTrainerGenerator
import matplotlib.pyplot as plt
import json


def main():
    generator_base = VGGTrainerGenerator() \
        .set_description("Check initial weight changes well pruning policy") \
        .set_device("0") \
        .set_output("experiments/retrain_with_reinit/result")

    trainer_original = generator_base.load_initial_weight_net("initial.model") \
        .log_report(filename="original.log") \
        .generate()

    trainer_with_pruning = generator_base.load_initial_weight_net("initial.model") \
        .apply_mask("initial.model", "final.model", 60) \
        .log_report("pruning.log") \
        .generate()

    trainer_with_reinit_and_pruning = generator_base.load_initial_weight_net("initial2.model") \
        .apply_mask("initial.model", "final.model", 60) \
        .log_report("reinit_pruning.log") \
        .generate()

    trainer_original.run()
    trainer_with_pruning.run()
    trainer_with_reinit_and_pruning.run()


def show_png():
    with open("experiments/retrain_with_reinit/result/original.log", "r") as fp:
        results = json.load(fp)

    log_size = int(len(results) / 3)
    res_original = results[0:log_size]
    res_pruning = results[log_size:log_size * 2]
    res_reinit_pruning = results[log_size * 2:log_size * 3]

    res_iteration = []

    res_original_train_loss_plot = []
    res_original_train_acc_plot = []
    res_original_test_loss_plot = []
    res_original_test_acc_plot = []

    res_pruning_train_loss_plot = []
    res_pruning_train_acc_plot = []
    res_pruning_test_loss_plot = []
    res_pruning_test_acc_plot = []

    res_reinit_pruning_train_loss_plot = []
    res_reinit_pruning_train_acc_plot = []
    res_reinit_pruning_test_loss_plot = []
    res_reinit_pruning_test_acc_plot = []

    iteration = "iteration"
    train_loss = "main/loss"
    train_acc = "main/accuracy"
    test_loss = "validation/main/loss"
    test_acc = "validation/main/accuracy"

    for i in range(log_size):
        res_original_i = res_original[i]
        res_pruning_i = res_pruning[i]
        res_reinit_pruning_i = res_reinit_pruning[i]

        res_iteration.append(res_original_i[iteration])

        res_original_train_loss_plot.append(res_original_i[train_loss])
        res_original_train_acc_plot.append(res_original_i[train_acc])
        res_original_test_loss_plot.append(res_original_i[test_loss])
        res_original_test_acc_plot.append(res_original_i[test_acc])

        res_pruning_train_loss_plot.append(res_pruning_i[train_loss])
        res_pruning_train_acc_plot.append(res_pruning_i[train_acc])
        res_pruning_test_loss_plot.append(res_pruning_i[test_loss])
        res_pruning_test_acc_plot.append(res_pruning_i[test_acc])

        res_reinit_pruning_train_loss_plot.append(res_reinit_pruning_i[train_loss])
        res_reinit_pruning_train_acc_plot.append(res_reinit_pruning_i[train_acc])
        res_reinit_pruning_test_loss_plot.append(res_reinit_pruning_i[test_loss])
        res_reinit_pruning_test_acc_plot.append(res_reinit_pruning_i[test_acc])

    plt.plot(res_iteration, res_original_train_loss_plot, label="original_train_loss")
    plt.plot(res_iteration, res_pruning_train_loss_plot, label="pruning_train_loss")
    plt.plot(res_iteration, res_reinit_pruning_train_loss_plot, label="reinit_pruning_train_loss")
    plt.legend()
    plt.title("Train Loss")
    plt.savefig("experiments/retrain_with_reinit/result/train_loss.png")

    plt.figure()

    plt.plot(res_iteration, res_original_train_acc_plot, label="original_train_accuracy")
    plt.plot(res_iteration, res_pruning_train_acc_plot, label="pruning_train_accuracy")
    plt.plot(res_iteration, res_reinit_pruning_train_acc_plot, label="reinit_pruning_train_accuracy")
    plt.legend()
    plt.title("Train Accuracy")
    plt.savefig("experiments/retrain_with_reinit/result/train_accuracy.png")

    plt.figure()

    plt.plot(res_iteration, res_original_test_loss_plot, label="original_test_loss")
    plt.plot(res_iteration, res_pruning_test_loss_plot, label="pruning_test_loss")
    plt.plot(res_iteration, res_reinit_pruning_test_loss_plot, label="reinit_pruning_test_loss")
    plt.legend()
    plt.title("Test Loss")
    plt.savefig("experiments/retrain_with_reinit/result/test_loss.png")

    plt.figure()

    plt.plot(res_iteration, res_original_test_acc_plot, label="original_test_accuracy")
    plt.plot(res_iteration, res_pruning_test_acc_plot, label="pruning_test_accuracy")
    plt.plot(res_iteration, res_reinit_pruning_test_acc_plot, label="reinit_pruning_test_accuracy")
    plt.legend()
    plt.title("Test Accuracy")
    plt.savefig("experiments/retrain_with_reinit/result/test_accuracy.png")


if __name__ == '__main__':
    show_png()
