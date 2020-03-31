import torch
import torch.nn as nn
import kornia
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import os
import numpy as np
import utils as utils
import torch.nn.functional as F


def downsample(x):
    x[:, :, 1::2, ::2] = x[:, :, ::2, ::2]
    x[:, :, ::2, 1::2] = x[:, :, ::2, ::2]
    x[:, :, 1::2, 1::2] = x[:, :, ::2, ::2]
    # x[:, :, ::2+1, ::2+1] = 0
    return x


def get_batch_params(x):
    batch_size = x.shape[0]
    bessel = (batch_size - 1) / batch_size
    mean = torch.mean(x, 0)
    std = torch.sqrt(torch.var(x, 0) * bessel + 1e-05)
    return mean, std


# x is the 'source' of downplaying, y is the 'target' of downplaying
def downplay(x, y, factor):
    idxs = (torch.sum(x, dim=1, keepdim=True) == 0).repeat(1,x.shape[1],1,1)
    y[idxs] = y[idxs] / factor
    return y


class MirrorLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(MirrorLayer, self).__init__()

        self.pad = nn.ReflectionPad2d(int((kernel-1)/2))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.conv(self.pad(x)))


class GaussianFilter(nn.Module):
    def __init__(self, sigma, channels):
        super(GaussianFilter, self).__init__()
        kernel = sigma*4+1
        self.pad = nn.ReflectionPad2d(int((kernel-1)/2))
        gfilter = kornia.filters.get_gaussian_kernel2d((kernel, kernel), (sigma, sigma)).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.conv = torch.nn.Conv2d(channels, channels, kernel, bias=False, groups=channels)
        self.conv.weight = nn.Parameter(gfilter)
        # self.conv.weight.required_grad = False

    def forward(self, x):
        return self.conv(self.pad(x))


class FilterAugment(nn.Module):
    def __init__(self, blurs):
        super(FilterAugment, self).__init__()

        self.filters = []
        for blur in blurs:
            self.filters.append(GaussianFilter(blur, 1))
        self.filters = nn.ModuleList(self.filters)

    def forward(self, x):
        augment = list()
        augment.append(x)
        for gfilter in self.filters:
            augment.append(gfilter(x))
        augment = torch.cat(augment, 1)
        return augment


class Estimator(nn.Module):
    def __init__(self, **kwargs):
        super(Estimator, self).__init__()
        self.start_epoch = 0
        self.config = kwargs

        self.augment = FilterAugment([4, 12, 48, 92])

        self.layers = list()
        for i in range(len(kwargs['dims'])):
            if i == 0:
                self.layers.append(MirrorLayer(5, kwargs['dims'][i], kwargs['kernels'][i]))
            else:
                self.layers.append(MirrorLayer(kwargs['dims'][i-1]+5, kwargs['dims'][i], kwargs['kernels'][i]))
        self.layers = nn.ModuleList(self.layers)

    def process(self, x):
        x = self.augment(x)
        y = self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            y = self.layers[i].forward(torch.cat([y, x], 1))
        return y

    def forward(self, **net_input):
        net_input['l_hat'] = self.process(net_input['nx'])
        return net_input

    def estimate(self, x):
        l_hat = self.process(x)
        return l_hat

    def save_model(self, path, filename):
        model = {
            'model': Estimator,
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(model, path + filename)

    @staticmethod
    def load_model(path, filename):
        checkpoint = torch.load(path + filename)
        model = checkpoint['model'](**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


class Denoiser(nn.Module):
    def __init__(self, **kwargs):
        super(Denoiser, self).__init__()
        self.start_epoch = 0
        self.config = kwargs
        self.layers = list()
        for i in range(len(kwargs['dims'])):
            if i == 0:
                self.layers.append(MirrorLayer(2, kwargs['dims'][i], kwargs['kernels'][i]))
            else:
                self.layers.append(MirrorLayer(kwargs['dims'][i-1]+2, kwargs['dims'][i], kwargs['kernels'][i]))
        self.layers = nn.ModuleList(self.layers)

    def process(self, x):
        y = self.layers[0].forward(x)
        for i in range(1, len(self.layers)):
            y = self.layers[i].forward(torch.cat([x, y], 1))
        return y

    def forward(self, **net_input):
        nxl = torch.cat([net_input['nx'], net_input['l']], 1)
        net_input['x_hat'] = self.process(nxl)
        return net_input

    def denoise(self, x, lam):
        nxl = torch.cat([x, lam], 1)
        y = self.process(nxl)
        return y

    # def load(self, apath, file='model_latest.pt', resume=-1):
    #     load_from = None
    #     kwargs = {}
    #     if resume == -1:
    #         load_from = torch.load(os.path.join(apath, file), **kwargs)
    #     if load_from:
    #         self.load_state_dict(load_from, strict=False)

    def save_model(self, path, filename):
        model = {
            'model': Denoiser,
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        torch.save(model, path + filename)

    @staticmethod
    def load_model(path, filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(path + filename)
        else:
            checkpoint = torch.load(path + filename, map_location='cpu')
        model = checkpoint['model'](**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


