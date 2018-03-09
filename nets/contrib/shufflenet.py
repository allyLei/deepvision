'''
No pretrained model of shufflenet.
Our own implementation.
'''

import math
import torch
import torch.nn as nn


class Shuffle(nn.Module):

    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C/g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Bottleneck(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        planes = out_planes / 4
        if in_planes == 24:
            conv1_groups = 1
        else:
            conv1_groups = groups

        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=1, groups=conv1_groups, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.shuffle1 = Shuffle(groups=conv1_groups)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=stride,
                               padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes,
                               kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU(inplace=True)

        if stride == 2:
            self.downsample = nn.AvgPool2d(3, stride=2, padding=1)


    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.shuffle1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if hasattr(self, 'downsample'):
            residual = self.downsample(residual)
            x = torch.cat([x, residual], 1)
        else:
            x = x + residual
        x = self.relu3(x)
        return x


class ShuffleNet(nn.Module):

    def __init__(self, planes, layers, groups,
                 depth_multiplier=1.0, num_classes=None):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = int(24 * depth_multiplier)
        planes = map(lambda x: int(x * depth_multiplier), planes)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes[0], layers[0], groups)
        self.layer2 = self._make_layer(planes[1], layers[1], groups)
        self.layer3 = self._make_layer(planes[2], layers[2], groups)

        if self.num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(planes[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, groups):
        layers = []
        layers.append(
                Bottleneck(self.in_planes, planes - self.in_planes, stride=2, groups=groups))
        self.in_planes = planes
        for i in range(1, blocks):
            layers.append(
                    Bottleneck(self.in_planes, planes, stride=1, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat_map = []
        x = self.layer1(x)
        feat_map.append(x)
        x = self.layer2(x)
        feat_map.append(x)
        x = self.layer3(x)
        feat_map.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, feat_map

def shufflenet4(**kwargs):
    planes = [272, 544, 1088]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=4, **kwargs)
    return model

def shufflenet3(**kwargs):
    planes = [240, 480, 960]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=3, **kwargs)
    return model

def shufflenet2(**kwargs):
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=2, **kwargs)
    return model

def shufflenet1(**kwargs):
    planes = [144, 288, 576]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=1, **kwargs)
    return model




