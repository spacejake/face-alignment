import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from face_alignment.models import ConvBlock, Bottleneck,  conv3x3, ResNetDepth, HourGlass

class DepthRegressor(ResNetDepth):
    '''
    ResNet-50
    '''
    def __init__(self, in_features=256, hidden_features=256,
                 block=Bottleneck, layers=[3, 4, 6, 3], num_classes=68):
        super(DepthRegressor, self).__init__(in_features, hidden_features,
                                             block, layers, num_classes)

    def _make_input_layer(self, in_features, hidden_features):
        return nn.Sequential(
            nn.Conv2d(in_features, hidden_features,
                      kernel_size=7, stride=1, padding=3,
                      bias=False), # 256x64x64
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer_in(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class FAN3D(nn.Module):

    def __init__(self, num_modules=1):
        super(FAN3D, self).__init__()
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

            downsample_seq = [
                conv3x3(256, 256),
                nn.BatchNorm2d(256),
            ]
            self.downsample_skip = nn.Sequential(*downsample_seq)

            self.depth_regressor = DepthRegressor()


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
            # tmp_out = self._modules['l' + str(i)](ll)
            tmp_out = torch.sigmoid(self._modules['l' + str(i)](ll))
            outputs.append(tmp_out)

            ll = self._modules['bl' + str(i)](ll)
            tmp_out_ = self._modules['al' + str(i)](tmp_out)

            if i < self.num_modules - 1:
                previous = previous + ll + tmp_out_

        x = self.downsample_skip(x) + ll + tmp_out_
        depth = self.depth_regressor(x)

        return outputs, depth
