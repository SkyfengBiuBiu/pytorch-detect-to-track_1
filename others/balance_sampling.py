from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler,WeightedRandomSampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16



# Create dummy data with class imbalance 99 to 1
numDataPoints = 1000
data_dim = 5
bs = 1
data = torch.randn(numDataPoints, data_dim)
target = torch.cat((torch.zeros(int(numDataPoints * 0.99)),
                    torch.ones(int(numDataPoints * 0.01))))
target=target.long()

print('target train 0/1: {}/{}'.format(
    (target == 0).sum(), (target == 1).sum()))

# Compute samples weight (each sample should get its own weight)
target1=torch.from_numpy(np.unique(target.numpy()))
class_sample_count = torch.Tensor(
    [(target == t).sum() for t in target1])
weight = 1. / class_sample_count.float()
samples_weight = torch.Tensor([weight[t] for t in target])

# Create sampler, dataset, loader
sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))
train_dataset = torch.utils.data.TensorDataset(data, target)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

# Iterate DataLoader and check class balance for each batch
a,b=0,0
for i, (x, y) in enumerate(train_loader):
    
    a+=(y == 0).sum()
    b+=(y == 1).sum()
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    print("x.shape {}, y.shape {}".format(x.shape, y.shape))
    
    
print("index 0/1: {}/{}".format(a,b))
    
    
    
    
    
    
    
    
    
    
    
    
    