"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division

import argparse

import chainer
from chainer import dataset
from chainer import datasets
from chainer import links as L
from chainer.training import extensions
from model_saver import save, load
from model import VggCNN6


def main():
    parser = argparse.ArgumentParser(description='Experiments of Lottery Ticket Hypothesis.')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='Device specifier. If non-negative integer, CuPy arrays with specified device id are used.'
                             'If negative integer, Numpy arrays are used.')
    parser.add_argument('--process', '-p', type=int, default=8, help='Number of parallel data loading processes.')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.0004, help='Learning rate.')
    parser.add_argument('--batch', '-b', type=int, default=48, help='Learning mini-batch size.')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number og sweeps over the dataset to train.')
    parser.add_argument('--initial_model', type=str, default=None,
                        help='Initial model with random weight.'
                             'If non-empty, --store_initial_model will be false.')
    parser.add_argument('--store_initial_model', action='store_true',
                        help='If initial model should be stored, set true, otherwise set false.'
                             'If false, --stored_initial_name is ignored.')
    parser.set_defaults(store_initial_model=False)
    parser.add_argument('--stored_initial_name', type=str, default='initial_vgg6.model',
                        help='Set the name of storing initial model with random_weight')
    parser.add_argument('--store_final_model', action='store_true',
                        help='If final(trained) model should be stored, set true, otherwise set false'
                             'If false, --stored_final_name is ignored.')
    parser.set_defaults(store_final_model=False)
    parser.add_argument('--stored_final_name', type=str, default='final_vgg6.model',
                        help='Set the name of storing initial model with random_weight')
    args = parser.parse_args()

    print('Train VGG-based 6-layers CNN.')
    print(f'# Minibatch-size: {args.batch}')
    print(f'# epoch: {args.epoch}')

    # Set up device.
    # If use GPU, set --device as non-negative number.
    # Otherwise, CPU will be used.
    print(f'Device: {args.device}')
    device = chainer.get_device(args.device)
    device.use()

    # Set up neural network.
    if args.initial_model:
        print(f'Load model: {args.initial_model} ...')
        net = load(args.initial_model)
        print('Model loaded.')
    else:
        print(f'Build new model...')
        net = VggCNN6()
        print(f'New model have built.')
        print(f'Store initial model: {args.store_initial_model}')
        if args.store_initial_model:
            save(args.stored_initial_name, net)
    print(f'Store final model: {args.store_final_model}')

    model = L.Classifier(net)
    model.to_device(device)

    # Set up dataset.
    train, test = datasets.get_cifar10()
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batch, n_processes=args.process
    )
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batch, repeat=False, n_processes=args.process
    )
    converter = dataset.concat_examples

    # In this experiment, Adam is used as optimizer.
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device
    )
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), 'result')

    interval = (300, 'iteration')

    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=device), trigger=interval)
    trainer.extend(extensions.LogReport(trigger=interval))
    trainer.extend(extensions.observe_lr(), trigger=interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

    if args.store_final_model:
        save(args.stored_final_name, net)


if __name__ == '__main__':
    main()
