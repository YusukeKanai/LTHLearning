"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import chainer

from chainer import dataset
from chainer import datasets
from chainer import iterators
from chainer import training
from chainer.training import extensions

from vgg import VggCNN6
from model_saver import save, load
import util


class VGGTrainer:
    def __init__(self, action):
        self.action = action

    def run(self):
        self.action()


class VGGTrainerGenerator:
    def __init__(self, debug=True):
        self.__debug = debug
        self.__description = None
        self.__net = None
        self.__batch_size = 48
        self.__epoch = 15
        self.__learning_rate = 0.00035
        self.__device_id = '-1'
        self.__save_initial_model_name = None
        self.__save_final_model_name = None
        self.__mask = None
        self.__output = 'result'
        self.__early_stop = False
        self.__interval = (150, 'iteration')
        self.__extensions = []

        self.train, self.test = datasets.get_cifar10()

    def set_description(self, description):
        if self.__debug:
            assert isinstance(description, str)
        self.__description = description
        return self

    def load_initial_weight_net(self, net):
        if self.__debug:
            assert isinstance(net, str)
        self.__net = load(net)
        return self

    def set_batch_size(self, batch_size):
        if self.__debug:
            assert isinstance(batch_size, int)
        self.__batch_size = batch_size
        return self

    def set_epoch(self, epoch):
        if self.__debug:
            assert isinstance(epoch, int)
        self.__epoch = epoch
        return self

    def set_learning_rate(self, learning_rate):
        if self.__debug:
            assert isinstance(learning_rate, float)
        self.__learning_rate = learning_rate
        return self

    def set_device(self, device_id):
        if self.__debug:
            assert isinstance(self.__device_id, str)
        self.__device_id = device_id
        return self

    def save_initial_net(self, name):
        self.__save_initial_model_name = name
        return self

    def save_final_net(self, name):
        self.__save_final_model_name = name
        return self

    def apply_mask(self, init_weight_net, final_weight_net, pruning_rate):
        self.__mask = util.create_masks(init_weight_net, final_weight_net, pruning_rate)
        return self

    def set_output(self, output):
        self.__output = output
        return self

    def apply_early_stop(self, on):
        self.__early_stop = on
        return self

    def log_report(self, filename="log", postprocess=None):
        self.__extensions.append(
            extensions.LogReport(trigger=self.__interval, filename=filename, postprocess=postprocess))
        return self

    def plot_report(self, y_keys, filename):
        self.__extensions.append(extensions.PlotReport(y_keys, trigger=self.__interval, filename=filename))
        return self

    def extensions(self, *extensions):
        self.__extensions.extend(extensions)
        return self

    def generate(self):
        print(self.__description)
        print(f'# Minibatch-size: {self.__batch_size}')
        print(f'# Epoch: {self.__epoch}')
        if not self.__net:
            self.__net = VggCNN6()

        # region : Setup device
        if int(self.__device_id) < 0:
            print('Run network with CPU.')
        else:
            print('Run network with GPU.')
        device = chainer.get_device(self.__device_id)
        device.use()
        # endregion

        if self.__save_initial_model_name:
            print(f'Save model before train: {self.__save_initial_model_name}')

        if self.__save_final_model_name:
            print(f'Save model after train: {self.__save_final_model_name}')

        if self.__save_initial_model_name:
            save(self.__save_initial_model_name, self.__net)

        model = chainer.links.Classifier(self.__net)

        optimizer = chainer.optimizers.Adam(alpha=self.__learning_rate)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.optimizer_hooks.WeightDecay(5e-4))

        train_iter = iterators.MultiprocessIterator(
            self.train, self.__batch_size, n_processes=8
        )
        updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=dataset.concat_examples,
                                                    device=device)

        trigger = (self.__epoch, 'epoch') if not self.__early_stop else \
            training.triggers.EarlyStoppingTrigger(monitor='validation/main/loss', check_trigger=self.__interval,
                                                   patients=10, max_trigger=(self.__epoch, 'epoch'))

        trainer = chainer.training.Trainer(updater, stop_trigger=trigger, out=self.__output)

        # Execute pruning at every iteration
        if self.__mask:
            @training.make_extension(trigger=(1, 'iteration'))
            def _pruner_extension(_):
                self.__net.prune_connection(self.__mask)

            # prune network at first.
            _pruner_extension(None)
            # prune network after back-propagation.
            trainer.extend(_pruner_extension)

        test_iter = iterators.MultiprocessIterator(self.test, self.__batch_size, repeat=False, n_processes=8)
        trainer.extend(extensions.Evaluator(test_iter, model, device=device), trigger=self.__interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy'
        ]), trigger=self.__interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        for extension in self.__extensions:
            trainer.extend(extension, trigger=self.__interval)

        def __vgg_trainer():
            trainer.run()
            if self.__save_final_model_name:
                save(self.__save_final_model_name)

        return VGGTrainer(__vgg_trainer)
