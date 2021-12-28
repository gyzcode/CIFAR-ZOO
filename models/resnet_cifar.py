import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

norm_mean, norm_var = 0.0, 1.0
epm = np.exp( 1. * .3)
ddecay = torch.pow(torch.tensor(0.96), epm)

count_num = 0
current_epoch = 0
isForTest = False

g_batch_ind = 0


cc_num =  11

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, compress_rate=[0.], compress_ratio=[0.]):
        super(ResBasicBlock, self).__init__()
        self.compress_rate = compress_rate
        self.compress_ratio = compress_ratio
        self.inplanes = inplanes
        self.planes = planes
        self.decay = torch.tensor(.9).cuda()

        self.conv1 = conv3x3(int(inplanes*compress_rate[0]), int(planes*compress_rate[1]), stride)
        self.conv1.cp_rate = compress_rate[1]
        self.bn1 = nn.BatchNorm2d(int(planes*compress_rate[1]))
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(int(planes*compress_rate[1]), int(planes*compress_rate[2]))
        self.conv2.cp_rate = compress_rate[2]
        self.bn2 = nn.BatchNorm2d(int(planes*compress_rate[2]))
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x, isTest=False):
        out = self.conv1(x)
        out = self.bn1(out)

        global count_num
        global ddecay
        global current_epoch
        global isForTest
        global g_batch_ind
        count_num += 1
        xb, xc, xh, xw = out.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if current_epoch+g_batch_ind > cc_num - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - cc_num))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[1]):, :, :] *= tddecay * (1.0+3e-1*torch.rand(att[:, int(xc*self.compress_ratio[1]):, :, :].size()).cuda())
            '''
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[1]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[1]):, :, :].size()).cuda() * 5e-2
            '''
        else:
            pass
        out = out.mul(att)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = self.relu2(out)

        count_num += 1
        xb, xc, xh, xw = out.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if current_epoch+g_batch_ind > cc_num - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - cc_num))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[2]):, :, :] *= tddecay * (1.0+3e-1*torch.rand(att[:, int(xc*self.compress_ratio[2]):, :, :].size()).cuda())
            '''
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[2]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[2]):, :, :].size()).cuda() * 5e-2
            '''
        else:
            pass
        '''
        tmp = False
        if np.random.randint(0, 1000) == 1:
            print(torch.mean(torch.mean(out[1,:,:,:],dim=1),dim=1))
            tmp = True
        '''
        out = out.mul(att)
        '''
        if tmp:
            print(torch.mean(torch.mean(out[1,:,:,:],dim=1),dim=1), ">>>>>>>>>>>>>>>>")
        '''


        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, covcfg, compress_rate, compress_ratio, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.compress_rate = compress_rate
        self.compress_ratio = compress_ratio
        self.num_layers = num_layers
        self.decay = torch.tensor(.9).cuda()

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, int(self.inplanes*self.compress_rate[0]), kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.cp_rate = compress_rate[0]

        self.bn1 = nn.BatchNorm2d(int(self.inplanes*self.compress_rate[0]))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1,
                                       compress_rate=compress_rate[0:2 * n + 1], compress_ratio=compress_ratio[0:2 * n + 1])
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2,
                                       compress_rate=compress_rate[2 * n:4 * n + 1], compress_ratio=compress_ratio[2 * n:4 * n + 1])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2,
                                       compress_rate=compress_rate[4 * n:6 * n + 1], compress_ratio=compress_ratio[4 * n:6 * n + 1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_layers == 110:
            self.linear = nn.Linear(int(64 * block.expansion*self.compress_rate[-1]), num_classes)
        else:
            self.fc = nn.Linear(int(64 * block.expansion*self.compress_rate[-1]), num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, compress_rate, compress_ratio):
        layers = []

        layers.append(block(self.inplanes, planes, stride, compress_rate=compress_rate[0:3], compress_ratio=compress_ratio[0:3]))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, compress_rate=compress_rate[2 * i:2 * i + 3], compress_ratio=compress_ratio[2 * i:2 * i + 3]))

        return nn.Sequential(*layers)

    def forward(self, x, epoch=0.0, batch_ind=0.0, hook_flag=False, isTest=False):
        #epm = np.exp( (epoch+batch_ind) * .3)
        #ddecay = torch.pow(self.decay, epm)
        global ddecay
        global count_num
        global current_epoch
        global isForTest
        global g_batch_ind
        isForTest = isTest
        current_epoch = epoch
        g_batch_ind = batch_ind
        ddecay = torch.pow(self.decay, 2*(epoch+batch_ind)) #1.0 - 0.08 * (epoch+batch_ind) 
        if ddecay < 5e-2:
            ddecay = 0.0
        if np.random.randint(0, 100) == 1:
            print(ddecay, "PPPPPPP")

        id_num = 0
        count_num = 1
        x = self.conv1(x)
        x = self.bn1(x)
        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if epoch + batch_ind > cc_num - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(epoch+batch_ind + count_num/10 - cc_num))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[id_num]):, :, :] *= tddecay * (1.0 + 3e-1*torch.rand(att[:, int(xc*self.compress_ratio[id_num]):, :, :].size()).cuda())
            '''
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[id_num]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[id_num]):, :, :].size()).cuda() * 5e-2
            '''
        else:
            pass
        x = x.mul(att)

        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        count_num += 1
        '''
        xb, xc = x.size()
        att = torch.ones(1, xc).cuda()
        if epoch > 13 - count_num / 10:
            att[:, int(xc*self.compress_ratio[-2]):] *= ddecay * (1.0 + 5e-1*torch.rand(att[:, int(xc*self.compress_ratio[-2]):].size()).cuda())
            if ddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[-2]):] = torch.rand(att[:, int(xc*self.compress_ratio[-2]):].size()).cuda() * 1e-1
                if np.random.randint(0, 100) == 1:
                    print(ddecay, "PPP             PPPP")
        else:
            pass
        x = x.mul(att)
        '''

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56(num_classes=10, rate=None, ratio=None):
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg, compress_rate=rate, compress_ratio=ratio)


def resnet_110(num_classes=10, rate=None, ratio=None):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg, compress_rate=rate, compress_ratio=ratio)

