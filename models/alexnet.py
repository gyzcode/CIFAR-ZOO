# -*-coding:utf-8-*-
import torch.nn as nn
import torch
import numpy as np

__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes, rate, ratio):
        super(AlexNet, self).__init__()
        self.rate = rate
        self.ratio = ratio
        self.decay = torch.tensor(.9).cuda()
        self.features = nn.Sequential(
            nn.Conv2d(3, int(64*rate), kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(int(64*rate), int(192*rate), kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(int(192*rate), int(384*rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(384*rate), int(256*rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256*rate), int(256*rate), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(int(256*rate), num_classes)

    def forward(self, x, epoch=0.0, batch_ind=0.0):
        epm = np.exp( (epoch+batch_ind) * 0.15)

        x = self.features[0](x)

        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
        x = x.mul(att)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)

        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
        x = x.mul(att)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)

        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
        x = x.mul(att)
        x = self.features[7](x)
        x = self.features[8](x)

        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
        x = x.mul(att)
        x = self.features[9](x)
        x = self.features[10](x)

        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        att[:, int(xc*self.ratio):, :, :] *= torch.pow(self.decay, epm)
        x = x.mul(att)

        x = self.features[11](x)
        x = self.features[12](x)

        x = x.view(x.size(0), -1)
        xb, xc = x.size()
        att = torch.ones(1, xc).cuda()
        att[:, int(xc*self.ratio):] *= torch.pow(self.decay, epm)
        x = x.mul(att)
        
        x = self.fc(x)
        return x


def alexnet(num_classes, rate=1.0, ratio=1.0):
    return AlexNet(num_classes=num_classes, rate=rate, ratio=ratio)
