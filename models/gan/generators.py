from models.gan.utils import *

"""
###########################################################################################
############################### GENERATORS Architectures ##################################
###########################################################################################
"""


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
