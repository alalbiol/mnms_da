import functools
from torch.nn import init
import torch.nn as nn
import torch

"""
#################################################################################
########################### Architecture operators ##############################
#################################################################################
"""


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0,
                    norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2, True))


def conv_spectral_lrelu(in_dim, out_dim, kernel_size, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias), eps=1e-6),
        nn.LeakyReLU(0.1, True)
    )


def conv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0,
                   norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
        norm_layer(out_dim), nn.ReLU(True))


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride=1, padding=0, output_padding=0,
                    norm_layer=nn.BatchNorm2d, bias=False, upsample=None):
    if upsample == "interpolation":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=1, padding=1, bias=bias),
            norm_layer(out_dim), nn.ReLU(True)
        )
    elif upsample == "deconvolution":
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding, bias=bias),
            norm_layer(out_dim), nn.ReLU(True)
        )
    else:
        assert False, f"Please specify a correct upsample. Unknown upsample '{upsample}'"
