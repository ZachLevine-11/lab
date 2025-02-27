import numpy as np
import torch
import torch.nn as nn
import os

class SubNetwork(nn.Module):
    def __init__(self, i, device, latent_size_factor = 100, inputSize = 1):
        super(SubNetwork, self).__init__()
        self.i = i
        self.inputSize = inputSize
        self.effective_latent_size_factor = max(self.inputSize//latent_size_factor, 1)
        ##the genetics binaries exclude SNPs whose rsid is "."
        ##makeSNPSets should exclude these by default from the SNP sets
        ##but in case it doesn't include another check
        ##Keep these layers small or we won't be able to load the network on disk
        self.layer1 = nn.Linear(in_features = self.inputSize, out_features = self.effective_latent_size_factor, dtype = torch.double)
        self.layer2 = nn.Linear(in_features= sels
        lf.effective_latent_size_factor, out_features=1, dtype = torch.double)
        self.device = device

##since all the data is loaded in from a file, the forward method only gets the ids of the people to pass through the nwtwork
    def forward(self, ids, x):
        ##Drop the trailing zeros that were padded
        ##All entries after the inputSize should be ignored
        ##Only consider the SNP channel of interest
        x = x[ids, :]
#         ##we need a 1 channel dimension for the convolution to work
#       x = x.reshape(-1, 1, self.inputSize)
        x = nn.functional.relu(self.layer1(x))
#        x = x.reshape([x.shape[0], x.shape[2]])
        x = self.layer2(x)
        return x

class fullNetwork(nn.Module):
    def __init__(self, device, cache_dir = "/net/mraid20/export/jasmine/zach/dl/cache/", latent_size_factor = 16, preload_cache = True, SnpSets = None):
        super(fullNetwork, self).__init__()
        self.latent_size_factor = latent_size_factor
        self.computed_cache = list(map(lambda fn: int(fn.split(".npy")[0]), os.listdir(cache_dir)))
        self.preload_cache = preload_cache
        self.lenCache = len(self.computed_cache)
        self.subNetworks = nn.ModuleDict()
        self.device = device
        self.loaded_cache = {}
        self.SnpSets = SnpSets
        ##make all subnetworks
        j = 0
        ##need order to match in overarching SNPset, so iterate over it
        for i in self.computed_cache:
            print("Creating subnetwork number: " + str(j) + "/" + str(self.lenCache))
            inputSize = len(self.SnpSets[list(self.SnpSets.keys())[int(i)]])
            self.subNetworks[str(i)] = SubNetwork(i, self.device, self.latent_size_factor, inputSize = inputSize).to(self.device)
            if self.preload_cache:
                self.loaded_cache[str(i)] = torch.from_numpy(np.load("/net/mraid20/export/jasmine/zach/dl/cache/" + str(i) + ".npy")).to(self.device)
            j += 1
        self.layer1 = nn.Linear(len(self.computed_cache), len(self.computed_cache)//2, dtype = torch.double)
        self.layer2 = nn.Linear(len(self.computed_cache)//2, len(self.computed_cache)//2, dtype = torch.double)
        self.layer3 = nn.Linear(len(self.computed_cache)//2, 1, dtype = torch.double)

    def forward(self, ids):
        subnetResults = []
        for i, subNet in self.subNetworks.items():
            if self.preload_cache:
                subnetResults.append(subNet(ids, self.loaded_cache[str(i)]))
            else:
                subnetResults.append(subNet(ids, torch.from_numpy(np.load("/net/mraid20/export/jasmine/zach/dl/cache/" + str(i) + ".npy")).to(self.device)))
        ##concat the results here
        subnetResults = torch.cat(subnetResults, axis = 1)
        x = self.layer1(subnetResults)
        x = self.layer2(x)
        x = self.layer3(x)
        return x.reshape(x.shape[0])
