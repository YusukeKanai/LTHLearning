# Reading a paper and Examination of "_The lottery ticket hypothesis_"

This repository contains the implementation to replicate the experiments "__Lottery Ticket Hypothesis__", and will be referd from [Qiita](https://qiita.com/): a technical knowledge sharing and collaboration site for programmers.

## References
[THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORK](https://openreview.net/pdf?id=rJl-b3RcF7)

## How to run

- train.py

```
usage: train.py [-h] [--device DEVICE] [--process PROCESS]
                [--learning_rate LEARNING_RATE] [--batch BATCH]
                [--epoch EPOCH] [--initial_model INITIAL_MODEL]
                [--store_initial_model]
                [--stored_initial_name STORED_INITIAL_NAME]
                [--store_final_model] [--stored_final_name STORED_FINAL_NAME]

Experiments of Lottery Ticket Hypothesis.

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE, -d DEVICE
                        Device specifier. If non-negative integer, CuPy arrays
                        with specified device id are used.If negative integer,
                        Numpy arrays are used.
  --process PROCESS, -p PROCESS
                        Number of parallel data loading processes.
  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        Learning rate.
  --batch BATCH, -b BATCH
                        Learning mini-batch size.
  --epoch EPOCH, -e EPOCH
                        Number og sweeps over the dataset to train.
  --initial_model INITIAL_MODEL
                        Initial model with random weight.If non-empty,
                        --store_initial_model will be false.
  --store_initial_model
                        If initial model should be stored, set true, otherwise
                        set false.If false, --stored_initial_name is ignored.
  --stored_initial_name STORED_INITIAL_NAME
                        Set the name of storing initial model with
                        random_weight
  --store_final_model   If final(trained) model should be stored, set true,
                        otherwise set falseIf false, --stored_final_name is
                        ignored.
  --stored_final_name STORED_FINAL_NAME
                        Set the name of storing initial model with
                        random_weight
```
- experiments.py
  - 1. Print accuracy graph from training log result.
  - 2. Print accuracy of initial random weight graph with various pruned weights. 


