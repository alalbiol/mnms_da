import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""
 RESNET34 UNET PRETRAINED ENCODER
"""


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True)
    )


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.interpolate(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class FPAv3(nn.Module):  # Custom  FPA for 224x224 input img sizes
    def __init__(self, input_dim, output_dim):
        super(FPAv3, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.interpolate(x_glob, scale_factor=x.shape[2], mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        # d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output


class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class ResUnet(nn.Module):
    def __init__(self, model_version, pretrained=True, num_classes=1, in_channels=1,
                 classification=False, add_scse=True, add_hypercols=True):
        super(ResUnet, self).__init__()

        self.classification = classification
        self.add_hypercols = add_hypercols

        if "resnet18" in model_version:
            self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif "resnet34" in model_version:
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        else:
            assert False, "Unknown model: {}".format(model_version)

        if pretrained:
            print("\n--- Frosted pretrained backbone! ---")
            for param in self.resnet.parameters():  # Frost model
                param.requires_grad = False

        self.resnet.conv1 = torch.nn.Conv2d(in_channels, 64, (7, 7), (2, 2), (3, 3), bias=False)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.encode2 = nn.Sequential(
            self.resnet.layer1, SCse(64) if add_scse else nn.Identity()
        )
        self.encode3 = nn.Sequential(
            self.resnet.layer2, SCse(128) if add_scse else nn.Identity()
        )
        self.encode4 = nn.Sequential(
            self.resnet.layer3, SCse(256) if add_scse else nn.Identity()
        )
        self.encode5 = nn.Sequential(
            self.resnet.layer4, SCse(512) if add_scse else nn.Identity()
        )

        self.center = nn.Sequential(
            FPAv3(512, 256), nn.MaxPool2d(2, 2)
        )

        if classification:
            self.linear = nn.Linear(256 * 1 * 1, num_classes)

        else:
            if add_hypercols:
                self.decode5 = Decoderv2(256, 512, 64)
                self.decode4 = Decoderv2(64, 256, 64)
                self.decode3 = Decoderv2(64, 128, 64)
                self.decode2 = Decoderv2(64, 64, 64)
                self.decode1 = Decoder(64, 32, 64)

                self.logit = nn.Sequential(nn.Conv2d(320, num_classes, kernel_size=3, padding=1))
            else:
                self.decode5 = Decoderv2(256, 512, 512)
                self.decode4 = Decoderv2(512, 256, 256)
                self.decode3 = Decoderv2(256, 128, 128)
                self.decode2 = Decoderv2(128, 64, 64)
                self.decode1 = Decoder(64, 32, 32)

                self.logit = nn.Sequential(nn.Conv2d(32, num_classes, kernel_size=3, padding=1))

    def forward(self, x):
        # x: (batch_size, 3, 224, 224)
        x = self.conv1(x)  # 64, 112, 112
        e2 = self.encode2(x)  # 64, 112, 112
        e3 = self.encode3(e2)  # 128, 56, 56
        e4 = self.encode4(e3)  # 256, 28, 28
        e5 = self.encode5(e4)  # 512, 14, 14

        f = self.center(e5)  # 256, 7, 7

        if self.classification:
            out = F.avg_pool2d(f, 7)  # (batch, 256, 8, 7) -> (batch, 256, 1, 1)
            out = out.view(out.size(0), -1)  # (batch, 256, 1, 1) -> (batch, 256)
            out = self.linear(out)
            return out

        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256

        if self.add_hypercols:
            f = torch.cat((d1,
                           F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
                           F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
                           F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
                           F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 256, 256
            logit = self.logit(f)  # 1, 256, 256

        else:
            logit = self.logit(d1)

        return logit


def resnet_model_selector(model_name, num_classes=1, classification=False, in_channels=3):
    if "resnet" in model_name and "unet" in model_name:
        if all(x not in model_name for x in ["scratch", "imagenet"]):
            assert False, "Please specify if scratch or pretrained on imagenet weights!"

        add_scse = True if "scse" in model_name else False
        add_hypercols = True if "hypercols" in model_name else False

        return ResUnet(
            model_name, pretrained=False if "scratch" in model_name else True, num_classes=num_classes,
            classification=classification, in_channels=in_channels, add_scse=add_scse, add_hypercols=add_hypercols
        ).cuda()

    else:
        assert False, "Unknown model selected!"
