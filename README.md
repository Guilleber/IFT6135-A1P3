# IFT6135-Assignment1

## Introduction

This project was made as an assignment for the class IFT6135 at University of Montreal. This is an application of convolutional neural networks to image classification (Cat vs Dog). This project have been realised in collaboration with 3 other students of the same class.

Hyper-parameters have been tuned using local random search arround some initial working parameters. These initial parameters are inspired from https://www.quora.com/What-is-the-VGG-neural-network .

## Install & prerequisites

To setup the project, please download the source code and create two additional folders "Models/" and "Data/". "Models/" will contain the model's parameters file once the training is completed. "Data/" should contain three sub-folders "train/", "valid/" and "test/" each one containing the corresponding dataset part splitted in two "Cat/" and "Dog/" folders.

The following instructions must be executed using Python 3 (tested on Python 3.6.2) with the following libraries:
- Pytorch 1.0 (other versions may not work)
- Numpy
- ImageIO
- Pillow

We applied a data augmentation technic to improve the results of our network. To do this we apply different random transformations to the images among:
- Vertical flip
- Random rotation from -30° to 30°
- Grayscale
- Random crop from 60% to 100% of the image size and resize

To apply our data augmentation method, please run the following commands:
'''bash
python3 augment.py ./Data/train/Cat/
python3 augment.py ./Data/train/Dog/
'''

## Train & Test

parameters.py

python3 train.py
--use-cuda
--epochs
--batch-size
--model-name

python3 test.py --model Models/<model_name>_e<epoch>.model
--use-cuda
--batch-size
