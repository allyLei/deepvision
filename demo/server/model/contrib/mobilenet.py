'''
Modified from github:
https://github.com/ruotianluo/pytorch-mobilenet-from-tf/blob/master/mobilenet.py
'''
import math
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True)
            )


class DepthwiseSeperableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise = self.conv_dw(in_channels, kernel_size, stride, dilation)
        self.pointwise = self.conv_pw(in_channels, out_channels, 1, stride=1)

    def conv_dw(self, in_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                          padding=1, groups=in_channels, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(in_channels, eps=0.001),
                nn.ReLU(inplace=True)
            )

    def conv_pw(self, in_channels, out_channels, kernel_size=1, stride=1):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.001),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):

    def __init__(self, num_classes=None, depth_multiplier=1.0):
        self.num_classes = num_classes
        super(MobileNet, self).__init__()
        planes = list(map(lambda x: int(x*depth_multiplier), [32, 64, 128, 256, 512, 1024]))
        self.conv1 = conv3x3(3, planes[0], 2)
        self.conv2 = DepthwiseSeperableConv(planes[0], planes[1],
                                   3, stride=1)
        self.layer1 = self._make_layer(planes[2], 2)
        self.layer2 = self._make_layer(planes[3], 2)
        self.layer3 = self._make_layer(planes[4], 6)
        self.layer4 = self._make_layer(planes[5], 2)

        if num_classes is not None:
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(planes[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        in_planes = planes // 2
        layers.append(DepthwiseSeperableConv(in_planes, planes, 3, stride=2))
        for i in range(1, blocks):
            layers.append(DepthwiseSeperableConv(planes, planes, 3, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        feat_map = []
        x = self.layer1(x)
        x = self.layer2(x)
        feat_map.append(x)
        x = self.layer3(x)
        feat_map.append(x)
        x = self.layer4(x)
        feat_map.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            #x = F.dropout(x, self.dropout, self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, feat_map


def mobilenet(**kwargs):
    model = MobileNet(depth_multiplier=1.0, **kwargs)
    return model

def mobilenet75(**kwargs):
    model = MobileNet(depth_multiplier=0.75, **kwargs)
    return model

def mobilenet50(**kwargs):
    model = MobileNet(depth_multiplier=0.5, **kwargs)
    return model

def mobilenet25(**kwargs):
    model = MobileNet(depth_multiplier=0.25, **kwargs)
    return model





