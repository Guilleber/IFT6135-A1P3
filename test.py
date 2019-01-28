import argparse
import os
import datetime
import json
import math
import numpy as np
import torch as t
from torch.optim import SGD
import batcher
import model
import parameters
import csv

def main():
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--model-name', default="model")

    args = parser.parse_args()
    params = parameters.params

    test_batch_loader = batcher.Batcher("./Data/test/")

    model = mdl.ConvNet(params)
    if args.use_cuda:
        model = model.cuda()

    learning_rate = params["learning_rate"]
    optimizer = SGD(model.parameters(), learning_rate)

    test_step = model.tester()
    img, pred = test(test_step)

    with open('./Results/sample_submission.csv', mode='w') as csv_file:
        fieldnames = ['id','label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in len(img):
            writer.writerow('id': img.split('/')[-1][:-4], 'label' : test_batch_loader.classes[pred[i]] )



    return


if __name__ == "__main__":
    main()
