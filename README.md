# README

> On the large batch training of CIFAR-10

## usage

```
usage: main.py [-h] [--lr LR] [--bs BS] [--epoch EPOCH] [--lr_decay LR_DECAY]
               [--warmup_epochs WARMUP_EPOCHS] [--opt OPT] [--log_dir LOG_DIR]
               [--psuedo PSUEDO_BATCH]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate                       
  --bs BS               training batch size                          
  --epoch EPOCH         epoch
  --lr_decay LR_DECAY   lr decay gamma for StepLR
  --warmup_epochs WARMUP_EPOCHS
                        warmup epochs
  --opt OPT             optimizer
  --log_dir LOG_DIR     log dir
  --psuedo  PSUEDO_BATCH  simulation of multi-worker training
```

- default values:
  - `lr`: 0.1
  - `bs`: 128
  - `lr_decay`: 0.1
  - `warmup_epochs`: 1, i.e. no warmup
  - `psuedo`: 0, i.e. dont simulate multi-worker training
  - `opt`: sgd
  - `epoch`: 200
  - `log_dir`: ./logs; no need to change

## reproduce of results

```shell
sh run.sh
```

## reproduce of graphs

see the `experiments_graph.ipynb`, the used information is dumped in `representatives` folder, which could be fully reproduced with `run.sh`.
 
