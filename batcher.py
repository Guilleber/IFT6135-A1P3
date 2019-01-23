import numpy as np
import json
import imageio
from os import listdir
import random
		
class Batcher:
    def __init__(self, path_to_data, shuffle=True):
        self.path = path_to_data
        self.classes = ['Cat', 'Dog']
        self.i = 0
        self.epoch = 0
        self.shuffle = shuffle
		
        if self.shuffle:
            self.create_file_list()
        return
		
    def create_file_list(self):
        list = []
        for cl in self.classes:
            list_cl = listdir(self.path + cl + '/')
            list += [self.path + cl + '/' + el for el in list_cl]
        random.shuffle(list)
        self.imlist = list
        return


    def next_batch(self, batch_size=32):
        input = []
        target = []
        while batch_size > 0 and self.i != len(self.imlist):
            batch_size -= 1
            im = imageio.imread(self.imlist[self.i])
            label = self.classes.index(self.imlist[self.i].split('.')[-2])
            self.i+=1
            if im.ndim == 2:
                im = np.repeat(np.expand_dims(im, axis=2), 3, axis=2)
            input.append(np.transpose(im, (2, 0, 1)))
            target.append(label)
        if self.i == len(self.imlist):
            self.epoch += 1
            self.i = 0
            if self.shuffle:
                self.create_file_list()
        return np.array(input), np.array(target)
		
		
    def next_test_batch(self, batch_size=32):
        input = []
        while batch_size > 0 and self.i != len(self.imlist):
            batch_size -= 1
            im = imageio.imread(self.imlist[self.i])
            self.i+=1
            batch_size-=1
            input.append(np.transpose(im, (2, 0, 1)))
        if self.i == len(self.imlist):
            self.epoch += 1
            self.i = 0
            if self.shuffle:
                self.create_file_list()
        return np.array(input)


    def eval(self, eval_fct, use_cuda=False):
        self.epoch = 0
        self.i = 0
        sum_loss = 0
        sum_acc = 0
        nb_loss = 0
        while self.epoch == 0:
            batch = self.next_batch(32)
            loss, acc = eval_fct(batch, use_cuda=use_cuda)
            sum_loss += loss
            sum_acc += acc
            nb_loss += 1
        return sum_loss/nb_loss, sum_acc/nb_loss
			
		
    def test(self, test_fct):
        self.epoch = 0
        self.i = 0
        out_list = np.array([])
        while epoch == 0:
            batch = self.next_test_batch(32)
            out = test_fct(batch)
            out_list = np.concatenate(out_list, out) #<---
        return self.imlist, out_list