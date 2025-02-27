import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import pandas as pd
from os import listdir
from GeneticsPipeline.helpers_genetic import read_status_table

class sublayer(Function):
    def __init__(self, mask):
        super(sublayer, self).__init__()
        self.mask = mask

    def forward(self, input, weight):
        self.save_for_backward(input, weight)
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)
        output = input.mm(extendWeights.t())
        return output

    def backward(self, grad_output):
        input, weight = self.saved_tensors()
        grad_input = grad_weight = None
        extendWeights = weight.clone()
        extendWeights.mul_(self.mask.data)

        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(extendWeights)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input)
            grad_weight.mul_(self.mask.data)
        return grad_input, grad_weight

class subLayer(nn.Module):
    def __init__(self, device, input_features, output_features):
        super(subLayer, self).__init__()
        self.device = device
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(data = torch.Tensor(self.output_features, self.input_features, requires_grad = True).cuda())
        self.mask = None.cuda()
        self.mask = nn.Parameter(self.mask, requires_grad=True)

    def forward(self, x):
        return sublayer(self.mask)(x, self.weight)

class efficientNetwork(nn.Module):
    def __init__(self, device, snpsets, person_cache_dir = "/net/mraid20/export/jasmine/zach/dl/person_cache/", snplist_fname = "/net/mraid20/export/jasmine/zach/dl/snplist.csv"):
        super(efficientNetwork, self).__init__()
        self.device = device
        self.people = listdir(person_cache_dir)
        self.snplist = list(pd.read_csv(snplist_fname)["0"])


    def forward(self, integer_ids):
        for i in integer_ids:
            person_bins = pd.read_csv("/net/mraid20/export/jasmine/zach/dl/person_cache/" + str(i) + ".csv")["0"]
            ##drop the dummy SNP
            person_bins = person_bins[1:len(person_bins)]
            person_bins.index = self.snplist
            ##person_bins go to subnets
