# ERAV1 Session 8 Assignment
The school of AI ERA V1 Course related assignments and Quiz of Session 8. We have tried different normalization technqiues on CIFAR10 dataset to analyze and understand the impact of them. Normalization techniques used are Batch,Layer and Group. Created a fully convolution network on pytorch with ability to tweak in between normalization techniques.

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
- Keep the parameter count less than 50000
- Try and add one layer to another (Use Skip connections)
- Max Epochs is 20 and achieve minimum of 70% accuracy across the Normalization techniques

## Modules
- ERAV1S8_Skipconnection.ipynb : BN/LN/GN techniques with experimental results.
- model.py : Python class file for Model creation across all Normalization.
- utils.py : Python class file for model training and evaluation.

## Analysis 
### Experiment Reults : 
1.Batch Normalization Results: \
Parameters: 49136 \
Best Train Accuracy: 77.92 \
Best Test Accuracy: 79.07 
Metrics: 
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/97487513-c98a-47b3-a70e-9604344cd4d4)
Misclassified Images :
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/7632f5ee-90e7-4a6b-9061-279be5d01f3d)


2.Layer Normalization Results: \
Parameters: 49136 \
Best Train Accuracy: 71.65 \
Best Test Accuracy: 72.93
Metrics: 
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/b17dd193-d3e6-4c96-bf8d-73b4061277ac)
Misclassified Images :
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/7305a680-f4a3-440c-98d8-711280f6a008)


3.Group Normalization Results: \
Parameters: 49136 \
Best Train Accuracy: 72.44 \
Best Test Accuracy: 73.68 
Metrics: 
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/47d1d086-5356-4b18-a8b1-7da8b49c1e5f)
Misclassified Images :
![image](https://github.com/prabhukirangit/ERAV1/assets/33514187/0844964d-2002-4363-a39e-6515f69045c7)


## Troubleshooting

## Authors
- Prabhukiran Ganapavarapu



