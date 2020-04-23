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


class MirrorConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(MirrorConv, self).__init__()
        self.pad = nn.ReflectionPad2d(int((kernel-1)/2))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel)

    def forward(self, x):
        return self.conv(self.pad(x))


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


class EncoderModule(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.nonlin1 = nn.PReLU()
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.nonlin2 = nn.PReLU()
        self.conv3 = nn.Conv2d(4, 6, 3)
        self.nonlin3 = nn.PReLU()
        self.conv4 = nn.Conv2d(6, 8, 3)
        self.nonlin4 = nn.PReLU()

    def forward(self, input):
        input = self.nonlin1(self.conv1(input))
        input = self.nonlin2(self.conv2(input))
        input = self.nonlin3(self.conv3(input))
        input = self.nonlin4(self.conv4(input))
        return input


class DecoderModule(nn.Module):
    def __init__(self, **kwargs):
        super(DecoderModule, self).__init__()
        self.deconv4 = nn.ConvTranspose2d(8, 6, 3)
        self.nonlin4 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose2d(6, 4, 3)
        self.nonlin3 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose2d(4, 2, 3)
        self.nonlin2 = nn.PReLU()
        self.deconv1 = nn.ConvTranspose2d(2, 1, 3)
        self.nonlin1 = nn.PReLU()

    def forward(self, input):
        input = self.nonlin4(self.deconv4(input))
        input = self.nonlin3(self.deconv3(input))
        input = self.nonlin2(self.deconv2(input))
        input = self.nonlin1(self.deconv1(input))
        return input


class AutoLambdaEstimator(nn.Module):
    def __init__(self, **kwargs):
        super(AutoLambdaEstimator, self).__init__()
        self.start_epoch = 0
        self.config = kwargs
        self.encoder = EncoderModule(**kwargs)
        self.decoder = DecoderModule(**kwargs)
        self.scale = nn.Parameter(torch.tensor([1]).float())

    def forward(self, **net_input):
        x = net_input['x']
        code = self.encoder(x + (torch.randn_like(x) + torch.rand_like(x)*x)/5)
        x_hat = self.decoder(code)*self.scale
        net_input['x_hat'] = x_hat
        return net_input

    def save_model(self, path, filename):
        model = {
            'model': AutoLambdaEstimator,
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


class PatternMatcher(nn.Module):
    def __init__(self, kernel):
        super(PatternMatcher, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel)
        self.conv2 = nn.Conv2d(10, 10, 1)
        self.conv3 = nn.Conv2d(10, 1, 1)
        self.nonlin = nn.PReLU()

    def forward(self, input):
        input = self.nonlin(self.conv1(input))
        input = self.nonlin(self.conv2(input))
        input = self.nonlin(self.conv3(input))
        return input


class EstimatorKernel(nn.Module):
    def __init__(self, kernel):
        super(EstimatorKernel, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel, bias=False)
        gfilter = kornia.filters.get_gaussian_kernel2d((kernel, kernel), (kernel/3, kernel/3)).unsqueeze(0).repeat(1, 1, 1, 1)
        self.conv.weight = nn.Parameter(gfilter, requires_grad=True)

    def normalize(self):
        with torch.no_grad():
            kernel = self.conv.weight
            kernel = torch.abs(kernel)
            norm = torch.sum(kernel)
            kernel = kernel / norm
            self.conv.weight = nn.Parameter(kernel, requires_grad=True)
            del self._parameters['conv']
            self.register_parameter('conv', self.conv)

    def forward(self, input):
        # self.normalize()
        return self.conv(input)


class EstimatorSubmodule(nn.Module):
    def __init__(self, kernel):
        super(EstimatorSubmodule, self).__init__()
        self.kernel = kernel

        self.pattern_matcher = PatternMatcher(self.kernel)
        self.lambda_estimator = EstimatorKernel(self.kernel)

    def forward(self, input, max_kernel):
        input_size = input.size(2)
        rm_size = int(((max_kernel-self.kernel)/2))
        x = torch.tensor(input)
        if torch.cuda.is_available():
            x = x.cuda()
        x = x[:, :, rm_size:(input_size-rm_size), rm_size:(input_size-rm_size)]
        match = self.pattern_matcher.forward(x)
        estimate = self.lambda_estimator.forward(x)
        return match, estimate


class FFN(nn.Module):
    def __init__(self, dims):
        super(FFN, self).__init__()
        self.layers = list()
        for i in range(len(dims)-1):
            self.layers.append(nn.Conv2d(dims[i], dims[i+1], 1))
            self.layers.append(nn.PReLU())
        self.layers = nn.ModuleList(self.layers)
        # self.combo = nn.Conv2d(dims[0], 1, 1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # temp = self.combo(input)
        for i in range(len(self.layers)):
            input = self.layers[i](input)
        # input = self.softmax(input + temp)
        return input


# class EstimatorModule(nn.Module):
#     def __init__(self, **kwargs):
#         super(EstimatorModule, self).__init__()
#         self.start_epoch = 0
#         self.config = kwargs
#         self.noise = 0
#         self.max_kernel = max(kwargs['kdims'])
#
#         self.softmax = nn.Softmax(dim=1)
#         self.filters = list()
#         for i in range(len(kwargs['kdims'])):
#             kernel = kwargs['kdims'][i]
#             for j in range(kwargs['nfilts'][i]):
#                 self.filters.append(EstimatorSubmodule(kernel))
#         self.nfilters = len(self.filters)
#         self.filters = nn.ModuleList(self.filters)
#
#         self.selector = FFN([self.nfilters, self.nfilters, self.nfilters])
#
#     def process(self, x):
#         # print(x.shape)
#         input_size = x.size(2)
#         rm_size = self.filters[0].kernel-1
#         matches = torch.zeros(x.size(0), self.nfilters, input_size-rm_size, input_size-rm_size)
#         estimates = torch.zeros(x.size(0), self.nfilters, input_size-rm_size, input_size-rm_size)
#         if torch.cuda.is_available():
#             matches = matches.cuda()
#             estimates = estimates.cuda()
#         for i in range(self.nfilters):
#             temp_m, temp_e = self.filters[i].forward(x, self.max_kernel)
#             matches[:, i, :, :] = temp_m[:, 0, :, :]
#             estimates[:, i, :, :] = temp_e[:, 0, :, :]
#         scores = self.softmax(self.selector(matches))
#         lambdas = torch.sum(scores * estimates, dim=1).unsqueeze(1)
#         return lambdas
#
#     def forward(self, **net_input):
#         x = torch.tensor(net_input['x'])
#         x_center = int((x.size(2) - 1) / 2)
#         x_prime = torch.tensor(x[:, 0, x_center, x_center])
#         # x[:, 0, x_center, x_center] = x_prime + torch.randn_like(x_prime) * self.noise
#         x[:, 0, x_center, x_center] = torch.randn_like(x_prime) * 100
#         net_input['x_prime'] = x_prime
#         net_input['estimate'] = self.process(x)
#         return net_input
#
#     def save_model(self, path, filename):
#         model = {
#             'model': EstimatorModule,
#             'config': self.config,
#             'state_dict': self.state_dict(),
#         }
#         torch.save(model, path + filename)
#
#     @staticmethod
#     def load_model(path, filename):
#         if torch.cuda.is_available():
#             checkpoint = torch.load(path + filename)
#         else:
#             checkpoint = torch.load(path + filename, map_location='cpu')
#         model = checkpoint['model'](**checkpoint['config'])
#         model.load_state_dict(checkpoint['state_dict'])
#         return model


class SS_Submodule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super(SS_Submodule, self).__init__()

        self.mconv = MirrorConv(in_dim, out_dim, kernel_size)
        self.nonlin = nn.PReLU()

    def forward(self, input):
        return self.nonlin(self.mconv(input))


class SelfSupervisedEstimator(nn.Module):
    def __init__(self, **kwargs):
        super(SelfSupervisedEstimator, self).__init__()
        self.start_epoch = 0
        self.config = kwargs

        self.local_layer1 = SS_Submodule(1, 16, 3, 1)
        self.local_layer2 = SS_Submodule(16, 16, 3, 1)
        self.globl_layer1 = SS_Submodule(1, 16, 31, 1)
        self.globl_layer2 = SS_Submodule(16, 16, 3, 1)

        self.integ_layer1 = SS_Submodule(32, 32, 3, 1)
        self.integ_layer2 = SS_Submodule(32, 32, 3, 1)
        self.integ_layer3 = SS_Submodule(32, 1, 3, 1)

    def process(self, x):
        local_inf = self.local_layer2(self.local_layer1(x))
        globl_inf = self.globl_layer2(self.globl_layer1(x))
        integ_inf = torch.cat([local_inf, globl_inf], 1)
        integ_inf = self.integ_layer1(integ_inf)
        integ_inf = self.integ_layer2(integ_inf)
        integ_inf = self.integ_layer3(integ_inf)
        return integ_inf

    def forward(self, **net_input):
        sample = torch.poisson(net_input['x'])
        l_hat = self.process(sample)
        net_input['l_hat'] = l_hat
        return net_input

    def save_model(self, path, filename):
        model = {
            'model': SelfSupervisedEstimator,
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


class EstimatorModule(nn.Module):
    def __init__(self, **kwargs):
        super(EstimatorModule, self).__init__()
        self.start_epoch = 0
        self.config = kwargs
        self.noise = 0
        self.max_kernel = max(kwargs['kdims'])

        self.conv1 = nn.Conv2d(1, 64, kwargs['kdims'][0])
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 1)
        self.conv5 = nn.Conv2d(64, 1, 1)
        self.nonlin = nn.PReLU()

    def process(self, x):
        x = self.nonlin(self.conv1(x))
        x = self.nonlin(self.conv2(x))
        x = self.nonlin(self.conv3(x))
        x = self.nonlin(self.conv4(x))
        lambdas = self.nonlin(self.conv5(x))
        return lambdas

    def forward(self, **net_input):
        x = torch.tensor(net_input['x'])
        x_center = int((x.size(2) - 1) / 2)
        x_prime = torch.tensor(x[:, 0, x_center, x_center])
        x[:, 0, x_center, x_center] = torch.randn_like(x_prime) * 100
        net_input['x_prime'] = x_prime
        net_input['estimate'] = self.process(x)
        return net_input

    def save_model(self, path, filename):
        model = {
            'model': EstimatorModule,
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


class LambdaEstimator(nn.Module):
    def __init__(self, **kwargs):
        super(LambdaEstimator, self).__init__()
        self.start_epoch = 0
        self.config = kwargs
        self.noise = kwargs['noise']

        self.layers = list()
        for i in range(len(kwargs['dims'])):
            if i == 0:
                self.layers.append(MirrorLayer(1, kwargs['dims'][i], kwargs['kernels'][i]))
            else:
                self.layers.append(MirrorLayer(kwargs['dims'][i - 1], kwargs['dims'][i], kwargs['kernels'][i]))
        self.layers = nn.ModuleList(self.layers)

    def process(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x

    def forward(self, **net_input):
        if self.noise != 0:
            nx = net_input['x'] + torch.randn_like(net_input['x'])*self.noise
        else:
            nx = net_input['x']
        net_input['l_hat'] = self.process(nx)
        net_input['l_hat_2'] = self.process(net_input['l_hat'])
        net_input['l_hat_3'] = self.process(net_input['l_hat_2'])
        return net_input

    def estimate(self, x):
        l_hat = self.process(x)
        return l_hat

    def save_model(self, path, filename):
        model = {
            'model': LambdaEstimator,
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


