# IFT6135-Assignment1-Part3

## Introduction

This project was made as an assignment for the class IFT6135 at University of Montreal. This is an application of convolutional neural networks to image classification (Cat vs Dog). This project have been realised in collaboration with 3 other students.

Hyper-parameters have been tuned using local random search arround some initial working parameters. These initial parameters are inspired from https://www.quora.com/What-is-the-VGG-neural-network. Tuned parameters are the number of convolutional layers, the kernel size, the size of output linear layers, the number of channels and the learning rate.

Our model best hyper-parameters are stored in the "parameters.py" file and will be used as the default configuration of the model.

## Install & prerequisites

To setup the project, please download the source code and create two additional folders "Models/" and "Data/". "Models/" will contain the model's parameters file once the training is completed. "Data/" should contain three sub-folders "train/", "valid/" and "test/" each one containing the corresponding dataset part splitted (except "test/") in two "Cat/" and "Dog/" folders. Images' names should be <number>.<Dog/Cat>.jpg for "train/" and "valid/" and <number>.jpg for "test/"

The following instructions must be executed using Python 3 (tested on Python 3.6.2) with the following libraries:
- Pytorch 1.0 (other versions may not work)
- Numpy
- ImageIO
- Pillow
- argparse
- csv

We applied a data augmentation technique to improve the results of our network. To do this we apply different random transformations to the images among:
- Vertical flip
- Random rotation from -30° to 30°
- Grayscale
- Random crop from 60% to 100% of the image size and resize
We create that way 7 new images for each image in our train dataset, thus multiplying by 8 the size of the dataset.

To apply our data augmentation method, please run the following commands:
```bash
python3 augment.py ./Data/train/Cat/
python3 augment.py ./Data/train/Dog/
```

## Train & Test

To train the model, please run the following command:

```bash
python3 train.py
```

Other parameters are:
- --use-cuda (default: False)
- --epochs (default: 8)
- --batch-size (default: 32)
- --model-name (default: model)

The script will display random batches scores during training as well as validation results at the end of each epoch. A new model parameters file will be create inside "Models/". "train.py" will also plot the loss and accuracy curves at the end of the training.

```bash
python3 test.py --model Models/<model_name>_acc<accuracy>_e<epoch>.model
```

Other parameters are:
- --use-cuda (dafault: False)
- --batch-size (default: 32)

"test.py" will create a new "Results/current_result.csv" file containing the classes for the images in "test/".
