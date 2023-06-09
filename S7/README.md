# ERAV1 Session 6 Assignment
The school of AI ERA V1 Course related assignments and Quiz of Session 6.

## Table of contents
- Prerequisites
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

## Modules
- ERAV1S7_CodeBlock1.ipynb : Experiment 1 of training and testing the CNN model to achieve the accuracy within the specified constraints.
- ERAV1S7_CodeBlock2.ipynb : Experiment 2 of training and testing the CNN model to achieve the accuracy within the specified constraints.
- ERAV1S7_CodeBlock3.ipynb : Experiment 3 of training and testing the CNN model to achieve the accuracy within the specified constraints.
- model.py : Python class file for Model creation across all expeiments.
- utils.py : Python class file for model training and evaluation.

## Analysis 
### Experiment 1 : 
1. Target \
Make the model lighter. Paramaters should be less than 8k.\
Add Batch-norm to increase model efficiency. 
2. Results: \
Parameters: 7556 \
Best Train Accuracy: 99.73 \
Best Test Accuracy: 99.26 
3. Analysis: \
Good and lighter model! \
We have started to see over-fitting and Even if the model is pushed further, it won't be able to get to 99.4 

### Experiment 2 : 
1. Target \
Make the model lighter. Paramaters should be less than 8k.\
Add Batch-norm to increase model efficiency. \
Add Dropout at each layer. \
Add GAP and Increase model capacity.
2. Results: \
Parameters: 7530 \
Best Train Accuracy: 99.33 \
Best Test Accuracy: 99.41 
3. Analysis: \
Model seems to be achieving the targeted accuracy. But we're not seeing 99.4 or more as often as we'd like. We can further improve it. \
The model is not over-fitting at all. \
Seeing image samples, we can see that we can add slight rotation.  

#### Experiment 3 : 
1. Target \
Make the model lighter. Paramaters should be less than 8k.\
Add Batch-norm to increase model efficiency. \
Add Dropout at each layer. \
Add GAP and Increase model capacity.\
Use Data augmentation techniques (Rotation between 5-7 degress) \
Add ReduceLROnPlateau Scheduler 
2. Results: \
Parameters: 7878 \
Best Train Accuracy: 99.34 \
Best Test Accuracy: 99.49 
3. Analysis: \
Model seems to be achieving the targeted accuracy. We are able to see 99.4 or more atleast twice in the experiment results (Exatly 3 times within 14 epochs)
The model is under-fitting now. This is fine, as we know we have made our train data harder.  \
The test accuracy is also up, which means our test data had few images which had transformation difference w.r.t. train dataset.

## Troubleshooting

## Authors
- Prabhukiran Ganapavarapu


