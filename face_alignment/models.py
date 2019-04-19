import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class FAN(nn.Module):

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            # if hg_module < self.num_modules - 1:
            self.add_module(
                'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                             256, kernel_size=1, stride=1, padding=0))
        input_skip_seq = [
            conv3x3(3, 128),
            nn.BatchNorm2d(128),
        ]
        self.input_skip = nn.Sequential(*input_skip_seq)

        downsample_seq = [
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
        ]
        self.downsample_layer = nn.Sequential(*downsample_seq)

        # output
        up_sequence = [
            ConvBlock(256, 256),
            Interpolate(size=(128,128), mode='bilinear'), # 64 -> 128
            conv3x3(256, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Interpolate(size=(256, 256), mode='bilinear'), # 128 -> 256
        ]
        self.up_layer = nn.Sequential(*up_sequence)

        output_sequence = [
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 68, kernel_size=1, stride=1, padding=0)
        ]
        self.output_layer = nn.Sequential(*output_sequence)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            ll = self._modules['bl' + str(i)](ll)
            tmp_out_ = self._modules['al' + str(i)](tmp_out)

            if i < self.num_modules - 1:
                previous = previous + ll + tmp_out_

        # x = self.downsample_layer(x)
        out = self.downsample_layer(x) + ll + tmp_out_
        out = self.up_layer(out)
        out = self.input_skip(input) + out
        out = self.output_layer(out)

        return out, outputs

class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#ResNet Patch Discriminator, based on SN-GAN  https://github.com/godisboy/SN-GAN/
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN = False, downsample=False):
        super(ResBlock, self).__init__()
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)


    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.LeakyReLU()]
        model += [conv3x3(in_channels, hidden_channels, padding=1)]
        model += [nn.LeakyReLU()]
        model += [conv3x3(hidden_channels, out_channels, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)


    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [conv3x3(in_channels, out_channels, padding=1)]
        model += [nn.LeakyReLU(0.2, True)]
        model += [conv3x3(out_channels, out_channels, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)


    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)


    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)

class ResPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, ndlayers=4, use_sigmoid=False):
        super(ResPatchDiscriminator, self).__init__()
        self.res_d = self.make_model(in_channels, ndf, ndlayers, use_sigmoid)

    def make_model(self, in_channels, ndf, ndlayers, use_sigmoid):
        model = []
        model += [OptimizedBlock(in_channels, ndf)]
        tndf = ndf
        downsample = True
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=downsample)]
            # downsample = not downsample
            tndf *= 2
        model += [nn.LeakyReLU(0.2, True)]

        # output 1 channel
        model += [ResBlock(tndf, 1, downsample=False)]

        if use_sigmoid:
            model += [nn.Sigmoid()]

        return nn.Sequential(*model)


    def forward(self, input):
        out = self.res_d(input)
        return out

class ResDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, ndlayers=4, use_sigmoid=False):
        super(ResDiscriminator, self).__init__()
        self.res_d = self.make_model(in_channels, ndf, ndlayers, use_sigmoid)
        self.fc_in_size = ndf*(2**ndlayers)

        out_seq = [nn.Linear(self.fc_in_size, 1)]

        if use_sigmoid:
            out_seq += [nn.Sigmoid()]

        self.fc = nn.Sequential(*out_seq)

        # self.fc = nn.Sequential(nn.Linear(self.fc_in_size, 1), nn.Sigmoid())

    def make_model(self, in_channels, ndf, ndlayers, use_sigmoid):
        model = []
        model += [OptimizedBlock(in_channels, ndf)]
        tndf = ndf
        downsample = True
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=downsample)]
            # downsample = not downsample
            tndf *= 2
        model += [nn.LeakyReLU(0.2, True)]

        return nn.Sequential(*model)


    def forward(self, input):
        out = self.res_d(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, self.fc_in_size)

        return self.fc(out)


 # Defines the PatchGAN/binary discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        #sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        #if use_sigmoid:
        #    sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        
        self.fc_in_size = ndf * nf_mult
        self.fc = nn.Sequential(nn.Linear(self.fc_in_size, 1), nn.Sigmoid())


    def forward(self, input):
        out = self.model(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, self.fc_in_size)
        return self.fc(out)


# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         return self.main(input)
