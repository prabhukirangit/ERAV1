# ERAV1 Session 9 Assignment
The school of AI ERA V1 Course related assignments and Quiz of Session 9. We have tried different normalization technqiues on CIFAR10 dataset to analyze and understand the impact of them. Normalization techniques used are Batch,Layer and Group. Created a fully convolution network on pytorch satysifying the constraints mentioned in the section below.

## Table of contents
- Prerequisites
- Constraints
- Modules
- Usage
- Troubleshooting
- Authors

## Prerequisites
This module requires the following dependencies to be installed prior.
- python
- torch
- torchsummary
- matplotlib
  
## Constraints
- Has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated - kernels here instead of MP or strided convolution, then 200pts extra!)
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Modules
- ERAV1S9.ipynb : Training notebook with the network defined as per above constraints.
- model.py : Python class file for Model creation.
- utils.py : Python class file for model training and evaluation.

## Analysis 
### Experiment Reults : 
Parameters: 194,176 \
Best Train Accuracy: 77.92 \
Best Test Accuracy: 85.07 \
Metrics: \



## Troubleshooting

## Authors
- Prabhukiran Ganapavarapu




