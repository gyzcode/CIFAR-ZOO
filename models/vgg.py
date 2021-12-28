# -*-coding:utf-8-*-
import torch.nn as nn
import torch
import numpy as np
__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',
          512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512,
          512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, rate=1.0, ratio=1.0):
        super(VGG, self).__init__()
        self.rate = rate
        self.ratio = ratio
        self.decay = torch.tensor(.9).cuda()
        self.features = features
        #self.classifier = nn.Linear(int(512*rate), num_classes)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(int(cfg[-2]*rate), int(cfg[-1]*rate))),
            ('norm1', nn.BatchNorm1d(int(cfg[-1]*rate))),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(int(cfg[-1]*rate), num_classes)),
        ]))
        self._initialize_weights()
        self.conv_list = con_list = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40]

    def forward(self, x, epoch=0.0, batch_ind=0.0):
        epm = np.exp( (epoch+batch_ind) * 0.15)
        
        for ii in range(44):
            x = self.features[ii](x)
            if ii in self.conv_list:
                xb, xc, xh, xw = x.size()
                att = torch.ones(1, xc, 1, 1).cuda()
                att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
                x = x.mul(att)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, rate=1.0):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nn.Conv2d(in_channels, int(v*rate), kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(int(in_channels*rate), int(v*rate), kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v*rate)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(num_classes):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)


def vgg13(num_classes):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes)


def vgg16(num_classes, rate=1.0, ratio=1.0):
    return VGG(make_layers(cfg['D'], batch_norm=True, rate=rate), num_classes, rate=rate, ratio=ratio)


def vgg19(num_classes, rate=1.0, ratio=1.0):
    return VGG(make_layers(cfg['E'], batch_norm=True, rate=rate), num_classes, rate=rate, ratio=ratio)
