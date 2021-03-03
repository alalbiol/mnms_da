import os
import matplotlib.pyplot as plt
import copy
import numpy as np

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


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3,
                                    norm_layer=norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


"""
###########################################################################################
############################### GENERATORS Architectures ##################################
###########################################################################################
"""


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, upsample=None):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            if upsample == "interpolation":
                upconv = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1)
                )
            elif upsample == "deconvolution":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            else:
                assert False, f"Please specify a correct upsample. Unknown upsample '{upsample}'"

            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if upsample == "interpolation":
                upconv = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(inner_nc, outer_nc, 4, stride=2, padding=1, bias=use_bias)
                )
            elif upsample == "deconvolution":
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            else:
                assert False, f"Please specify a correct upsample. Unknown upsample '{upsample}'"

            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            if upsample == "interpolation":
                upconv = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1, bias=use_bias)
                )
            elif upsample == "deconvolution":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            else:
                assert False, f"Please specify a correct upsample. Unknown upsample '{upsample}'"

            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, upsample="interpolation"):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True, upsample=upsample
        )
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout, upsample=upsample
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer, upsample=upsample
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer, upsample=upsample
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer, upsample=upsample
        )
        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
            norm_layer=norm_layer, upsample=upsample
        )
        self.unet_model = unet_block

    def forward(self, x):
        return self.unet_model(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True,
                 num_blocks=6, upsample="interpolation"):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                     conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                     conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [
            dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias, upsample=upsample),
            dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias, upsample=upsample),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)


def define_Gen(
        input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0],
        checkpoint="", upsample="interpolation"
):
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        gen_net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=9, upsample=upsample
        )
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6, upsample=upsample
        )
    elif netG == 'unet_128':
        gen_net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, upsample=upsample
        )
    elif netG == 'unet_256':
        gen_net = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, upsample=upsample
        )
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    model = init_network(gen_net, gpu_ids)

    if checkpoint != "":
        print("Loaded model from checkpoint: {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint)["generator"])

    return model


"""
###############################################################################################
############################### DISCRIMINATORS Architectures ##################################
###############################################################################################
"""


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            dis_model += [conv_norm_lrelu(
                ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                norm_layer=norm_layer, padding=1, bias=use_bias
            )]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        dis_model += [conv_norm_lrelu(
            ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
            norm_layer=norm_layer, padding=1, bias=use_bias
        )]
        dis_model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, x):
        return self.dis_model(x)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, x):
        return self.dis_model(x)


def define_Dis(input_nc, ndf, netD, n_layers_D=3, norm='batch', gpu_ids=[0], checkpoint=""):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    if netD == 'n_layers':
        dis_net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_bias=use_bias)
    elif netD == 'pixel':
        dis_net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    model = init_network(dis_net, gpu_ids)

    if checkpoint != "":
        print("Loaded model from checkpoint: {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint))

    return model


"""
################################################################
########################### Other ##############################
################################################################
"""


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class SampleFromPool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


def get_random_labels(vol_x_original_label, available_labels):
    """
    vol_x_original_label -> tensor([0,1,1,2,3,0])
    available_labels -> [0,1,2] (num available vendors, ej. A-B-C)
    returns -> Different random labels tensor([1,2,0,1,1,1]) within available_labels
    """
    res = []
    for l in vol_x_original_label:
        res.append(
            np.random.choice([x for x in available_labels if x != l], 1)[0]
        )
    return torch.from_numpy(np.array(res))


def labels2rfield(labels, shape):
    # vol_label_u has shape [batch, channels, receptive_field, receptive_field], to be able to multiply
    # with random labels, we have to transform list labels shape [batch] to [batch, 1, 1, 1]
    # eg. labels -> [0,1,0,2,0]
    labels = labels.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    labels = torch.ones(shape).to(labels.device) * labels
    return labels


def plot_save_generated(original_img, generated_img, original_img_mask, generated_img_mask, save_dir, img_id):
    import warnings
    warnings.filterwarnings('ignore')

    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 6))
    plt.subplots_adjust(wspace=0.005, hspace=0.2, right=0.5)

    ax1[0].axis('off')
    ax1[1].axis('off')
    ax2[0].axis('off')
    ax2[1].axis('off')

    ax1[0].imshow(original_img, cmap="gray")
    ax1[0].set_title("Original Image")

    ax1[1].imshow(original_img_mask, cmap="gray")
    ax1[1].set_title("Original - Mask")

    ax2[0].imshow(generated_img, cmap="gray")
    ax2[0].set_title("Generated Image")

    ax2[1].imshow(generated_img_mask, cmap="gray")
    ax2[1].set_title("Generated - Mask")

    pred_filename = os.path.join(
        save_dir,
        f"generated_{img_id}.png",
    )
    plt.savefig(pred_filename, dpi=200, bbox_inches='tight')
    plt.close()
