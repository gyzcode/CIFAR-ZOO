
import math
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]


class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None, rate=1.0, ratio=1.0):
        super(VGG, self).__init__()
        self.features = nn.Sequential()

        if cfg is None:
            cfg = defaultcfg

        self.relucfg = relucfg
        self.covcfg = convcfg
        self.decay = torch.tensor(.9).cuda()

        self.rate = rate
        self.ratio = ratio
        self.features = self.make_layers(cfg[:-1], True, rate)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(int(cfg[-3]*rate[-3]), int(cfg[-2]*rate[-2]))),
            ('norm1', nn.BatchNorm1d(int(cfg[-2]*rate[-2]))),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(int(cfg[-2]*rate[-2]), num_classes)),
        ]))

        if init_weights:
            self._initialize_weights()
        
        self.conv_list = list( np.array([0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]) + 1 )

    def make_layers(self, cfg, batch_norm=True, rate=None):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        index_num = 0

        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                '''
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2d.cp_rate = rate[cnt]
                cnt += 1

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                '''
                if in_channels == 3:
                    conv2d = nn.Conv2d(in_channels, int(v*rate[index_num]), kernel_size=3, padding=1)
                else:
                    conv2d = nn.Conv2d(int(in_channels*rate[index_num-1]), int(v*rate[index_num]), kernel_size=3, padding=1)
                
                layers.add_module('conv%d' % i, conv2d)
                if batch_norm:
                    #layers += [conv2d, nn.BatchNorm2d(int(v*rate)), nn.ReLU(inplace=True)]
                    layers.add_module('norm%d' % i, nn.BatchNorm2d(int(v*rate[index_num])))

                if len(rate) > 1:
                    index_num += 1
                
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))

                in_channels = v

        return layers
    '''
    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    '''

    def forward(self, x, epoch=0.0, batch_ind=0.0, hook_flag=False, isForTest=False):
        #epm = np.exp( (epoch+batch_ind) * .3)
        ddecay = torch.pow(self.decay, 2*(epoch+batch_ind))
        if ddecay < 5e-2:
            ddecay = 0

        if np.random.randint(0, 100) == 1:
            print(ddecay, 2*(epoch+batch_ind), epoch, batch_ind, "PPPPPPP")

        id_num = 0
        for ii in range(43):
            x = self.features[ii](x)
            if ii in self.conv_list:
                if hook_flag:
                    xb, xc, xh, xw = x.size()
                    att = torch.ones(1, xc, 1, 1).cuda()
                    if epoch+batch_ind > 5 - ii / 10:
                        #tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - 5))
                        att[:, int(xc*self.ratio[id_num])+1:, :, :] *= ddecay * (1.0+3e-1*torch.rand(att[:, int(xc*self.ratio[id_num])+1:, :, :].size()).cuda()) #* torch.pow(self.decay, ii/43)

                        if ddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                            att[:, int(xc*self.ratio[id_num])+1:, :, :] = torch.rand(att[:, int(xc*self.ratio[id_num])+1:, :, :].size()).cuda() * 5e-2

                    id_num += 1
                    x = x.mul(att)
                
                if np.random.randint(0, 1000) == 1:
                    pass
                    #print(x)
                    #print(xc, int(xc*self.ratio[id_num])+1, "{{{{{{{&&&&&&&&&&&&&&&&&&&}}}}}}}")

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier.linear1(x)
        x = self.classifier.norm1(x)
    
        if hook_flag:
            xb, xc = x.size()
            att = torch.ones(1, xc).cuda()
            att[:, int(xc*self.ratio[id_num])+1:] *= ddecay * (1.0+3e-1*torch.rand(att[:, int(xc*self.ratio[id_num])+1:].size()).cuda()) 
            if ddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.ratio[id_num])+1:] = torch.rand(att[:, int(xc*self.ratio[id_num])+1:].size()).cuda() * 5e-2

            x = x.mul(att)

        x = self.classifier.relu1(x)
        x = self.classifier.linear2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_16_bn(num_classes=10, rate=1.0, ratio=1.0):
    return VGG(num_classes=10, rate=rate, ratio=ratio)

