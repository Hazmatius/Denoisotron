import kornia
import torch
import torch.nn as nn


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


class SS_Submodule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(SS_Submodule, self).__init__()

        self.mconv = MirrorConv(in_dim, out_dim, kernel_size)
        self.nonlin = nn.PReLU()

    def forward(self, input):
        return self.nonlin(self.mconv(input))


class Distiller(nn.Module):
    def __init__(self, **kwargs):
        super(Distiller, self).__init__()

        self.config = kwargs
        self.start_epoch = 0

        N = kwargs['dim']
        self.encoder_mat = nn.Linear(N, N, bias=False)
        self.encoder_mat.weight.data.copy_(torch.eye(N))
        self.decoder_mat = nn.Linear(N, N, bias=False)
        self.decoder_mat.weight.data.copy_(torch.eye(N))

    def forward(self, **net_input):
        self.encoder_mat.weight.data.clamp_(max=1)
        self.decoder_mat.weight.data.clamp_(min=0, max=1)

        x = net_input['x']
        y = self.encoder_mat(x)
        x_hat = self.decoder_mat(y)
        net_input['y'] = y
        net_input['x_hat'] = x_hat

        return net_input

    def save_model(self, path, filename):
        model = {
            'model': Distiller,
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


class SelfSupervisedEstimator(nn.Module):
    def __init__(self, **kwargs):
        super(SelfSupervisedEstimator, self).__init__()
        self.start_epoch = 0
        self.config = kwargs

        self.local_layer1 = SS_Submodule(1, 16, 3)
        self.local_layer2 = SS_Submodule(16, 16, 3)
        self.globl_layer1 = SS_Submodule(1, 16, 31)
        self.globl_layer2 = SS_Submodule(16, 16, 3)

        self.integ_layer1 = SS_Submodule(32, 32, 3)
        self.integ_layer2 = SS_Submodule(32, 32, 3)
        self.integ_layer3 = SS_Submodule(32, 1, 3)

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


class Denoiser(nn.Module):
    def __init__(self, **kwargs):
        super(Denoiser, self).__init__()
        self.start_epoch = 0
        self.config = kwargs
        self.layers = list()
        for i in range(len(kwargs['dims'])):
            if i == 0:
                self.layers.append(MirrorConv(2, kwargs['dims'][i], kwargs['kernels'][i]))
                self.layers.append(nn.PReLU())
            else:
                self.layers.append(MirrorConv(kwargs['dims'][i-1]+2, kwargs['dims'][i], kwargs['kernels'][i]))
                self.layers.append(nn.PReLU())
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


