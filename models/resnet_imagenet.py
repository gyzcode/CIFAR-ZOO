import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


norm_mean, norm_var = 1.0, 0.1
epm = np.exp( 1. * .3)
ddecay = torch.pow(torch.tensor(0.96), epm)

block_id = 0
count_num = 0

current_epoch = 0
isForTest = False
g_batch_ind = 0

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cp_rate=[0.], cp_ratio=[0.], tmp_name=None):
        super(ResBottleneck, self).__init__()
        self.decay = torch.tensor(.9).cuda()
        self.compress_ratio = cp_ratio
        self.conv1 = nn.Conv2d(int(inplanes*cp_rate[0]), int(planes*cp_rate[1]), kernel_size=1, bias=False)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(int(planes*cp_rate[1]))
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(int(planes*cp_rate[1]), int(planes*cp_rate[2]), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes*cp_rate[2]))
        self.conv2.cp_rate = cp_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(int(planes*cp_rate[2]), int(planes * self.expansion*cp_rate[3]), kernel_size=1, bias=False)
        self.conv3.cp_rate = cp_rate[2]
        self.conv3.tmp_name = tmp_name

        self.bn3 = nn.BatchNorm2d(int(planes * self.expansion*cp_rate[3]))
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        global block_id

        print(self.compress_ratio, "LLLLLLLLLLLLLLLLLLLLLL")

        self.att1 = torch.nn.Parameter(torch.ones(1, int(planes*cp_rate[1]), 1, 1).cuda())
        self.att2 = torch.nn.Parameter(torch.ones(1, int(planes*cp_rate[2]), 1, 1).cuda())
        self.att3 = torch.nn.Parameter(torch.ones(1, int(planes * self.expansion*cp_rate[3]), 1, 1).cuda())
        if self.downsample is not None:
            self.att4 = torch.nn.Parameter(torch.ones(1, int(planes * self.expansion*cp_rate[3]), 1, 1).cuda())
            self.att4.requires_grad = False
            self.register_parameter("att4"+str(block_id), self.att4)
            
        self.att1.requires_grad = False
        self.att2.requires_grad = False
        self.att3.requires_grad = False

        self.register_parameter("att1"+str(block_id), self.att1)
        self.register_parameter("att2"+str(block_id), self.att2)
        self.register_parameter("att3"+str(block_id), self.att3)

        block_id += 1
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        global current_epoch
        global g_batch_ind
        global count_num

        count_num += 1
        xb, xc, xh, xw = out.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if current_epoch + g_batch_ind > 6 - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - 6))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[0]):, :, :] *= tddecay * (1.0 + 3e-1*torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda())
            
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[0]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda() * 5e-2
        else:
            pass
        
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        count_num += 1
        xb, xc, xh, xw = out.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if current_epoch + g_batch_ind > 6 - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - 6))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[0]):, :, :] *= tddecay * (1.0 + 3e-1*torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda())
            
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[0]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda() * 5e-2
        else:
            pass

        out = out.mul(att)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        out = self.relu3(out)

        count_num += 1
        xb, xc, xh, xw = out.size()
        att = torch.ones(1, xc, 1, 1).cuda()
        if current_epoch + g_batch_ind > 6 - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(current_epoch+g_batch_ind + count_num/10 - 6))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[0]):, :, :] *= tddecay * (1.0 + 3e-1*torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda())
            
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[0]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[0]):, :, :].size()).cuda() * 5e-2
            
        else:
            pass
        
        out = out.mul(att)

        return out


class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, covcfg=None, cp_rate=None, cp_ratio=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.covcfg = covcfg
        self.compress_rate = cp_rate
        self.compress_ratio = cp_ratio
        compress_rate = cp_rate
        compress_ratio = cp_ratio
        self.num_blocks = num_blocks
        self.decay = torch.tensor(.9).cuda()

        self.conv1 = nn.Conv2d(3, int(64*cp_rate[0]), kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = 'conv1'
        self.bn1 = nn.BatchNorm2d(int(64*cp_rate[0]))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.att1 = torch.nn.Parameter(torch.ones(1, int(64*cp_rate[0]), 1, 1).cuda())
        self.att2 = torch.nn.Parameter(torch.ones(1, 2048, 1, 1).cuda())
        self.att1.requires_grad = False
        self.att2.requires_grad = False

        self.register_parameter("att1", self.att1)
        self.register_parameter("att2", self.att2)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       cp_rate=compress_rate[0:3*num_blocks[0]+2], cp_ratio=compress_ratio[0:3*num_blocks[0]+2],
                                       tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+1:3*num_blocks[0]+3*num_blocks[1]+3],cp_ratio=compress_ratio[3*num_blocks[0]+1:3*num_blocks[0]+3*num_blocks[1]+3],
                                       tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+2:3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4],cp_ratio=compress_ratio[3*num_blocks[0]+3*num_blocks[1]+2:3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4],
                                       tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+3:],cp_ratio=compress_ratio[3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+3:],
                                       tmp_name='layer4')

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(int(512 * block.expansion*cp_rate[-1]), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, cp_rate, cp_ratio, tmp_name):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_short = nn.Conv2d(int(self.inplanes*cp_rate[0]), int(planes * block.expansion*cp_rate[1]),
                                   kernel_size=1, stride=stride, bias=False)
            conv_short.cp_rate = cp_rate[0]
            conv_short.tmp_name = tmp_name + '_shortcut'
            downsample = nn.Sequential(
                conv_short,
                nn.BatchNorm2d(int(planes * block.expansion*cp_rate[1]),
            )
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cp_rate=cp_rate[0:4], cp_ratio=cp_ratio[0:4],
                            tmp_name=tmp_name + '_block' + str(1)))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cp_rate=cp_rate[3 * i:3 * i + 4], cp_ratio=cp_ratio[3 * i:3 * i + 4],
                                tmp_name=tmp_name + '_block' + str(i + 1)))

        return nn.Sequential(*layers)

    def forward(self, x, epoch=0.0, batch_ind=0.0, hook_flag=False, forTest=False):
        
        ddecay = torch.pow(self.decay, 2*(epoch+batch_ind))
        if np.random.randint(0, 100) == 1:
            print(ddecay, "PPPPPPP")

        if ddecay < 1e-1:
            ddecay=  0.0

        global count_num
        global current_epoch
        global g_batch_ind
        global isForTest
        current_epoch = epoch
        g_batch_ind = batch_ind
        isForTest = forTest
        count_num = 1
        
        id_num = 0

        x = self.conv1(x)
        x = self.bn1(x)
        xb, xc, xh, xw = x.size()
        att = torch.ones(1, xc, 1, 1).cuda()

        count_num = 0
        if epoch + batch_ind > 6 - count_num / 10:
            tddecay = torch.pow(self.decay, 2*(epoch+batch_ind + count_num/10 - 6))
            if tddecay < 5e-2:
                tddecay = 0.0
            att[:, int(xc*self.compress_ratio[id_num]):, :, :] *= tddecay * (1.0 + 3e-1*torch.rand(att[:, int(xc*self.compress_ratio[id_num]):, :, :].size()).cuda())
            
            if tddecay < 5e-2 and not isForTest and np.random.randint(0, 100) > 50:
                att[:, int(xc*self.compress_ratio[id_num]):, :, :] = torch.rand(att[:, int(xc*self.compress_ratio[id_num]):, :, :].size()).cuda() * 5e-2
        else:
            pass

        count_num += 1
        
        x = x.mul(att)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        # 2048 x 7 x 7
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_50(num_classes=10, rate=None, ratio=None):
    cov_cfg = [(3*i + 3) for i in range(3*3 + 1 + 4*3 + 1 + 6*3 + 1 + 3*3 + 1 + 1)]
    model = ResNet(ResBottleneck, [3, 4, 6, 3], covcfg=cov_cfg, cp_rate=rate, cp_ratio=ratio)
    return model
