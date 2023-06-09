# ERAV1 Session 6 Assignment
The school of AI ERA V1 Course related assignments and Quiz of Session 6.

## Table of contents
- Prerequisites
- Part 1
- Part 2
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

## Part 1

Backpropagation of a sample network performed in excel using chain rule and weight updates using GD.

<img width="931" alt="image" src="https://github.com/prabhukirangit/ERAV1/assets/33514187/8001277a-cc4f-43e8-88a2-fe2d5b868db3">

Loss grpah wiht different learning rates. Given the nature and size of dataset, smaller learning rates less than 1 took larger epochs or iterations to reduce loss compartive to rates >= 1. With learning rate >=1, the network seems to converge quicker and becomes plateau.

<img width="825" alt="image" src="https://github.com/prabhukirangit/ERAV1/assets/33514187/3ca83eed-d80d-4da2-b6d1-26036eae91b4">

## Part 2
## Modules
- ERAV1-S6.ipynb : Main notebook file to load the data, train and test the model.
- model.py : Python class file for Model creation, training.
- utils.py : Python class file for model evaluation.

## Troubleshooting

## Authors
- Prabhukiran Ganapavarapu

