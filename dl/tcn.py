import torch
from torch import nn as nn
from torch import optim
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from torch.nn import functional as F
import math

class small_proj(nn.Module):
    def __init__(self, dim = 232):
        super().__init__()
        self.act = nn.GELU()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x, skipLinear = False):
        x = x.reshape(x.size(0), -1)
        if skipLinear:
            return x
        x =  self.linear1(x)
        x = self.act(x)
        x =  self.linear2(x)
        x = F.normalize(x.view(x.size(0), x.size(1)), dim=1, p=2)
        return x

##https://arxiv.org/pdf/2104.04569
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.act =  nn.GELU()
        self.maxpool_res = nn.MaxPool1d(4)
        self.conv1a = nn.Conv1d(in_channels = 12, out_channels = 64, kernel_size = 16, stride = 1)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 16, stride = 1)
        self.bn1b = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)
        self.conv1c = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 16, stride = 1, padding  = 2)
        ##
        self.conv1d_res1 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 16, stride = 1)
        ##
        self.bn1c = nn.BatchNorm1d(128)
        self.conv2a = nn.Conv1d(in_channels = 128, out_channels = 196, kernel_size = 16, stride = 1)
        self.bn2a = nn.BatchNorm1d(196)
        self.conv2b = nn.Conv1d(in_channels = 196, out_channels = 196, kernel_size = 16, stride = 1, padding = 2)
        self.bn2b = nn.BatchNorm1d(196)
        ##
        self.conv1d_res_2 = nn.Conv1d(in_channels = 128, out_channels = 196, kernel_size = 16, stride = 1)
        ##
        self.conv2c = nn.Conv1d(in_channels = 196, out_channels = 256, kernel_size = 16, stride = 1)
        self.bn2c = nn.BatchNorm1d(256)
        self.conv3a = nn.Conv1d(in_channels = 256, out_channels = 392, kernel_size = 16, stride = 1, padding = 2)
        self.bn3a = nn.BatchNorm1d(392)
        ##
        self.conv1d_res_3 = nn.Conv1d(in_channels = 196, out_channels = 392, kernel_size = 16, stride = 1)
        ##
        self.conv3b = nn.Conv1d(in_channels = 392, out_channels = 768, kernel_size = 1, stride = 1)
        self.bn3b = nn.BatchNorm1d(768)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.act(x)
        z = x
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv1c(x) + self.conv1d_res1(self.maxpool_res(z))
        z = x
        x = self.bn1c(x)
        x = self.act(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2b(x) + self.conv1d_res_2(self.maxpool_res(z))
        z = x
        x = self.bn2b(x)
        x = self.act(x)
        x = self.conv2c(x)
        x = self.bn2c(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv3a(x) +  self.conv1d_res_3(self.maxpool_res(z))
        x = self.bn3a(x)
        x = self.act(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x

class transformerEncoder(nn.Module):
    def __init__(self, emb_dim, device):
        super().__init__()
        self.emb_token = torch.nn.Parameter(torch.randn(1, 1, emb_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(emb_dim, nhead = 12, batch_first=False, activation = "gelu", dim_feedforward = 3072)
        self.encoder_stack_one_eleven = nn.TransformerEncoder(self.encoder_layer, num_layers = 11)
        self.encoder_stack_twelve = nn.TransformerEncoder(self.encoder_layer, num_layers = 1)
        self.small_proj = small_proj(emb_dim)
        self.emb_token = nn.Parameter(torch.randn(1, 1, emb_dim, device = device), requires_grad = True)
        self.pos_emb = nn.Parameter(torch.randn(43, 1, emb_dim, device = device), requires_grad = True)

    def forward(self, x, skipLinear = False, stop_after_eleven = False):
        x = x.permute(2,0,1)
        x = torch.cat([self.emb_token.repeat(1, x.shape[1], 1,), x]) ##repeat the CLS token along the batch dimension to add it to each batch
        x = x + self.pos_emb.repeat(1, x.shape[1], 1,) ##learn positional embeddings
        x = self.encoder_stack_one_eleven(x)
        if stop_after_eleven: return x ##just for visualization of attention maps
        x = self.encoder_stack_twelve(x)
        x = x[0, :, :]
        x = self.small_proj(x, skipLinear = skipLinear)
        return x

class both(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.time_conv = encoder()
        self.transformer = transformerEncoder(768, device)

    def forward(self, x, skipLinear=False, stop_after_eleven = False):
        x = self.time_conv(x)
        x = self.transformer(x, skipLinear, stop_after_eleven)
        return x

    def get_activations_gradient(self):
        return self.time_conv.gradients

    def get_activations(self, x):
        return  self.time_conv(x)

def peakLoss(reconst, orig, alpha, beta, gamma, delta = 1, theta = 1):
    def obj(x, y):
        return nn.L1Loss()(x,y) #MSE
    loss = 0
    for person in range(orig.shape[0]):
        if theta > 0:
            for channel in range(12):
                reconst_stft = torch.stft(reconst[person, channel, :], n_fft = 10000, return_complex=True)
                orig_stft = torch.stft(orig[person, channel, :], n_fft = 10000, return_complex=True)
                loss += theta*obj(reconst_stft, orig_stft)
        if alpha > 0:
            for channel in range(12):
                ##penalize errors higher on ECG peaks of the original signal
                height = int(torch.std(orig[person, channel, :].cpu().detach()))*2
                peaks_orig = find_peaks(orig[person, channel, :].cpu().detach(), prominence = height)[0] ##findpeaks returns many things, we just want the indices of the peak#
                if len(peaks_orig) > 0:
                    loss += alpha*obj(reconst[person, channel, peaks_orig], orig[person, channel, peaks_orig])
                        ##now repeat for minima
                    min_peaks_orig = find_peaks(-1*orig[person, channel, :].cpu().detach(), prominence=height)[0]  ##findpeaks returns many things, we just want the indices of the peak#
                    loss += alpha * obj(reconst[person, channel, min_peaks_orig], orig[person, channel, min_peaks_orig])
                          ##do some moment matching
        for channel in range(12):
            loss += gamma*(torch.std(reconst[person, channel,:]) - torch.std(orig[person, channel,:]))**2
            loss += beta*(torch.mean(reconst[person, channel,:]) - torch.mean(orig[person, channel,:]))**2
    loss += delta*obj(orig, reconst)
    return loss

class SmallPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.la = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.la(x)
        return x