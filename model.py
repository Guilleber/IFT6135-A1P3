import numpy as np
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable	
	
	
class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()
        self.params = params
		
        self.Convs = nn.ModuleList()
        Conv = nn.Conv2d(3, self.params["channel_out"][0], self.params["kernel_size"][0], self.params["stride"][0], self.params["padding"][0])
        self.Convs.append(Conv)
        if self.params["nb_conv_layers"] != 1:
            for i in range(1,self.params["nb_conv_layers"]):
                Conv = nn.Conv2d(self.params["channel_out"][i-1], self.params["channel_out"][i], self.params["kernel_size"][i], self.params["stride"][i], self.params["padding"][i])
                self.Convs.append(Conv)
				
        self.Pools = nn.ModuleList()
        for i in range(self.params["nb_conv_layers"]):
            Pool = nn.MaxPool2d((self.params["pool_kernel_size"][i], self.params["pool_kernel_size"][i]))
            self.Pools.append(Pool)
		
        W = self.params["W_in"]
        H = self.params["H_in"]
        for i in range(self.params["nb_conv_layers"]):
            W = out_dim_conv(W, self.params["padding"][i], 1, self.params["kernel_size"][i], self.params["stride"][i])
            W = out_dim_pool(W, self.params["pool_kernel_size"][i])
            H = out_dim_conv(H, self.params["padding"][i], 1, self.params["kernel_size"][i], self.params["stride"][i])
            H = out_dim_pool(H, self.params["pool_kernel_size"][i])
			
        self.OutLayer = nn.Linear(H*W*self.params["channel_out"][-1], 2)


    def forward(self, input_batch):
        batch_size = input_batch.size()[0]
        for i in range(self.params["nb_conv_layers"]):
            input_batch = F.relu(self.Convs[i](input_batch))
            input_batch = self.Pools[i](input_batch)
        preactivation = self.OutLayer(input_batch.contiguous().view(batch_size, -1))
        return preactivation
		
		
    def trainer(self, optimizer):
        def train_step(batch, use_cuda=False):
            self.train()
            input_batch = batch[0]
            target_batch = batch[1]
            if use_cuda:
                input_batch = Variable(t.from_numpy(input_batch)).float().cuda()
                target_batch = Variable(t.from_numpy(target_batch)).long().cuda()
            else:
                input_batch = Variable(t.from_numpy(input_batch)).float()
                target_batch = Variable(t.from_numpy(target_batch)).long()
            optimizer.zero_grad()
            output_preac_batch = self(input_batch)
            loss = F.cross_entropy(output_preac_batch, target_batch)
            loss.backward()
            optimizer.step()
			
            return float(loss.data.cpu().numpy())
        return train_step
		

    def validator(self):
        def valid_step(batch, use_cuda=False):
            self.eval()
            input_batch = batch[0]
            target_batch = batch[1]
            if use_cuda:
                input_batch = Variable(t.from_numpy(input_batch)).float().cuda()
                target_batch = Variable(t.from_numpy(target_batch)).long().cuda()
            else:
                input_batch = Variable(t.from_numpy(input_batch)).float()
                target_batch = Variable(t.from_numpy(target_batch)).long()
            output_preac_batch = self(input_batch)
            loss = F.cross_entropy(output_preac_batch, target_batch)
            pred = [int(np.argmax(output_preac)) for output_preac in output_preac_batch.data.cpu().numpy()]
            acc = sum([1 if target_batch[i] == pred[i] else 0 for i in range(len(pred))])/len(pred)
			
            return float(loss.data.cpu().numpy()), acc
        return valid_step
		
		
    def tester(self):
        def test_step(batch, use_cuda=False):	
            self.eval()
            input_batch = batch[0]
            if use_cuda:
                input_batch = Variable(t.from_numpy(input_batch)).float().cuda()
            else:
                input_batch = Variable(t.from_numpy(input_batch)).float()
            output_preac_batch = self(input_batch)
            pred = [int(np.argmax(output_preac)) for output_preac in output_preac_batch.data.cpu().numpy()]
			
            return pred
        return test_step
			

def out_dim_conv(in_dim, padding, dilation, kernel_size, stride):
    return math.floor((in_dim + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1)

def out_dim_pool(in_dim, kernel_size):
    return math.floor((in_dim - (kernel_size - 1) - 1)/kernel_size + 1)