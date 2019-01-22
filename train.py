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

def main():
    parser = argparse.ArgumentParser(description='IFT6135')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--model-name', default=model)
    
    args = parser.parse_args()
    params = parameters.params
    
    train_batch_loader = batcher.Batcher("./data/train/")
    valid_batch_loader = batcher.Batcher("./data/valid/")
    
    model = model.ConvNet(params)
    if args.use_cuda:
        model = model.cuda()
    
    learning_rate = params["learning_rate"]
    optimizer = SGD(model.parameters(), learning_rate)
    
    train_step = model.trainer(optimizer)
    valid_step = model.validator()
    
    tracking_valid_loss = []
    tracking_train_loss = []
    current_epoch = 0
    
    while current_epoch < args.epochs:
        iteration = 0
        while current_epoch == train_batch_loader.epoch:
            batch = train_batch_loader.next_batch(batch_size=args.batch_size)
            if iteration%50 == 0:
                loss = train_step(batch, use_cuda=args.use_cuda)
                tracking_train_loss.append(loss)
                print("Epoch: " + str(current_epoch + 1) + ", It: " + str(iteration + 1) + ", Loss: " + str(loss))
            else:
                _ = train_step(batch, use_cuda=args.use_cuda)
            iteration += 1
        current_epoch += 1
        t.save(model.state_dict(), "./models/" + args.model_name + "_e" + str(current_epoch) + ".model")
        loss_valid, acc_valid = valid_batch_loader.eval(valid_step)
        tracking_valid_loss.append(loss_valid)
        print('\n')
        print("***VALIDATION***")
        print("Epoch: " + str(current_epoch) + ", Loss: " + str(loss_valid) + ", Acc: " + str(acc_valid))
        print("****************")
        print('\n')
        if current_epoch >= 5:
            if current_epoch >= 8:
                learning_rate = learning_rate/2
                optimizer = SGD(model.parameters(), learning_rate)
                train_step = model.trainer(optimizer)
            else:
                if tracking_valid_loss[-2] >= tracking_valid_loss[-1]:
                    learning_rate = learning_rate/2
                    optimizer = SGD(model.parameters(), learning_rate)
                    train_step = model.trainer(optimizer)
    return
    
  
if __name__ == "__main__":
    main()