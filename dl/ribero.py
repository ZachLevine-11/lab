import torch
from torch import nn
from tcn import small_proj

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
        self.conv3a = nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 16, stride = 1, padding = 2)
        self.bn3a = nn.BatchNorm1d(256)
        ##
        self.conv1d_res_3 = nn.Conv1d(in_channels = 196, out_channels = 256, kernel_size = 16, stride = 1)
        ##
        self.conv3b = nn.Conv1d(in_channels = 256, out_channels = 320, kernel_size = 16, stride = 1)
        self.bn3b = nn.BatchNorm1d(320)
        self.gradients = None

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
        x = self.act(x)
        x = torch.mean(x, dim = 2).squeeze()
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x

class both(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_conv = encoder()
        self.small_proj = small_proj(320)

    def forward(self, x, skipLinear=False, stop_after_eleven = False):
        x = self.time_conv(x)
        x = self.small_proj(x, skipLinear = skipLinear)
        return x

    def get_activations_gradient(self):
        return self.time_conv.gradients

    def get_activations(self, x):
        return  self.time_conv(x)
