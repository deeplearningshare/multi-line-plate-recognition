Multi-line license plate recognition
======
A simple code for creating licence plate images and train e2e network based on a paper named an End-to-End Neural Network for Multi-line License Plate Recognition of  ICPR2018

****
	
|Author|deeplearningshare|
|---|---
|E-mail|1162450005@qq.com


****
## Requirements
* tensorflow-gpu 1.9.0
* python 3.6.5
* some common packages like numpy and so on.

## Quick start
* run create_train_data.py to create plate image and corresponding labels. This repository also contains the plate generator and can generate thousands of plates.
* then run createImage_data.py to create pp.npy file
* reset the train data path and run train_nn.py to train your model.

## Attention
I only submit the method of creating the singal line plate ,I will submit the method of creating the double line plate later

## test result
singal line:
train :20000 test:2000 acc:99.98% 

double line:
train:20000 test:2000 acc:99.95%
all the test result base on Artificially generated license plate


