# Replicating the experiments of "_The lottery ticket hypothesis_"

This repository contains the implementation to replicate the experiments "__Lottery Ticket Hypothesis__", and will be referd from [Qiita](https://qiita.com/): a technical knowledge sharing and collaboration site for programmers.

## References
[THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORK](https://openreview.net/pdf?id=rJl-b3RcF7)

## Requirements

```
pip install -r requirements.txt
```

## Experiments

### Dataset

Cifar-10

### Network

VGG-based 6 Convolutional Netowrok and 2 Full Connected Network.The trained model reached 84.82% accuracy on testing set.
Detail parameter configs and performance described in \[to be linked!\].

### Accuracy of initial weighted pruned model.

11.60% accuracy with original data, on the other hand, a pruned accuracy reached 19.10%.
However it is found that even not so much rate of pruning can decrease accuracy.
Detail result described in \[to be linked!\].

### Retrained result

T.B.C

## How to use

### Check performance of original network

In root directory of this repository,
```
python experiments\original_perfoemance\original_performance.py
```
### Check the accuracy of initial weighted model with different pruning rate

In root directory of this repository,
```
python experiments\original_perfoemance\original_performance.py
```
### Check the accuracy of retrained model with different pruning rate

In root directory of this repository,
```
python experiments\original_perfoemance\retrain.py
```

