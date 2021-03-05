from .discriminator import *
from .generators import *
import models.gan.resnet as studio


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
    elif netG == "studio_gen":
        model = studio.Generator(z_dim=256, img_size=256, g_conv_dim=32, g_spectral_norm=False, attention=True,
                 attention_after_nth_gen_block=2, conditional_bn=False, num_classes=input_nc,
                 initialize=False, mixed_precision=False, activation_fn="ReLU").cuda()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    if netG != "studio_gen":
        model = init_network(gen_net, gpu_ids)

    if checkpoint != "":
        print("Loaded model from checkpoint: {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint)["generator"])

    return model


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
