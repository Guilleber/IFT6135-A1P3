import argparse
import os
from os import listdir
import datetime
import json
import math
import numpy as np
import torch as t
import PIL
import torchvision

def main():
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--dir')
    
    args = parser.parse_args()
    imlist = listdir(args.dir)
    new_images = 7
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomGrayscale(p=0.1),
        torchvision.transforms.RandomHorizontalFlip(p=4.0/7.0),
        torchvision.transforms.RandomRotation(degrees=30),
        torchvision.transforms.RandomResizedCrop(64, scale=(0.6,1.0))
    ])
    
    for imname in imlist:
        im = PIL.Image.open(args.dir + imname)
        for i in range(new_images):
            transform(im).save(args.dir + imname[:-4] + '.' + str(i) + ".jpg")
        im.close()
    
    return
    
  
if __name__ == "__main__":
    main()