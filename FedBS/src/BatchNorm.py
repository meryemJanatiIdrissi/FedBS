import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn
from torch.autograd import Variable
from torch import nn
import pickle as pkl
import os
import random
import string
from options import args_parser



class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5,
                 affine=True, track_running_stats=True):
        
        
        args = args_parser()
        if args.ma == 'cma':
            momentum = 0.1
        else:
            momentum = None
        
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        args = args_parser()
        
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                # print("================== TRACKED ===============", self.num_batches_tracked)
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                if args.ma == 'sma':
                    self.running_mean = 1/2 *  (mean + self.running_mean)
                    self.running_var = 1/2 *  (var + self.running_var) + torch.pow(self.running_mean, 2)
                else:
                    self.running_mean = exponential_average_factor * mean \
                                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.running_var

            mean = (1 - exponential_average_factor) * mean \
                                    +  exponential_average_factor * self.running_mean
            var = (1 - exponential_average_factor) * var \
                                   + exponential_average_factor * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # the old one need to be checked

        return input


