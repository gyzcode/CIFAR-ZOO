# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os

from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

from easydict import EasyDict
from models import *
from data import imagenet

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()

mydecay = torch.tensor(.9).cuda()
epm = np.exp( .0 * 3.)
ddecay = torch.pow(mydecay, epm)

def hook_layers_vgg16(net, my_ratio=0.8):
    ratio = my_ratio[0]
    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)+1
        ch2 = int(grad_input[1].size(1)*ratio)+1
        grad_weight = grad_input[1].clone()
        grad_weight[0:ch, :, :, :] *= 1.0 
        grad_weight[ch:, :, :, :] *= ddecay
        grad_bias = grad_input[2].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0

        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)+1
        ch2 = int(grad_input[1].size(1)*ratio2)+1
        grad_weight = grad_input[1].clone()
        grad_weight[ch:, :, :, :] *= ddecay
        grad_weight[:, ch2:, :, :] *= ddecay
        grad_weight[0:ch, 0:ch2, :, :] *= 1.0
        grad_bias = grad_input[2].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)+1
        ch2 = int(grad_input[2].size(1)*ratio2)+1
        grad_weight = grad_input[2].clone()
        grad_weight[ch:, :] *= ddecay
        grad_weight[:, ch2:] *= ddecay
        grad_weight[0:ch, 0:ch2] *= 1.0 
        grad_bias = grad_input[0].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_bias, grad_input[1], grad_weight)

    def hook_bn(module, grad_input, grad_output):
        #print(module.running_mean, module.running_var, ">>>>>")
        ch = int(grad_input[2].size(0)*ratio)+1
        grad_weight = grad_input[2].clone()
        grad_weight[ch:] *= ddecay
        grad_bias = grad_input[1].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_input[0], grad_bias, grad_weight)
    

    def hook_f3(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)+1
        ch2 = int(grad_input[2].size(1)*ratio2)+1
        grad_weight = grad_input[2].clone()
        grad_weight[:, ch2:] *= ddecay
        grad_weight[:, 0:ch2] *= 1.0 
        #grad_bias[0:ch] = grad_input[0][0:ch]
        grad_bias = grad_input[0].clone()
    
        return (grad_bias, grad_input[1], grad_weight)
    
    con_list = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    '''
    for iii in range(100):
        if isinstance(net.features[iii], nn.Conv2d):
            print(iii, "PPPPPPPPPPPPPPPP")
    '''
    ratio = my_ratio[0]
    net.features[0].register_backward_hook(hook_f0)
    net.features[1].register_backward_hook(hook_bn)
    id_num = 1
    for ii in con_list:
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.features[ii].register_backward_hook(hook_f1)
        net.features[ii+1].register_backward_hook(hook_bn)

    
    fc_layer = net.classifier
    fc_layer.linear1.register_backward_hook(hook_f2)
    fc_layer.norm1.register_backward_hook(hook_bn)
    id_num += 1
    ratio = my_ratio[id_num]
    fc_layer.linear2.register_backward_hook(hook_f3)


def hook_layers_resnet56(net, my_ratio=0.8):
    ratio = my_ratio[0]

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = grad_input[1].clone()
        grad_weight[:ch, :, :, :] *= 1.0 
        grad_weight[ch:, :, :, :] *= ddecay
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None

        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio2)
        grad_weight = grad_input[1].clone()
        grad_weight[ch:, :, :, :] *= ddecay
        grad_weight[:, ch2:, :, :] *= ddecay
        grad_weight[0:ch, 0:ch2, :, :] *= 1.0
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[ch:, :] *= ddecay
        grad_weight[:, ch2:] *= ddecay
        grad_weight[0:ch, 0:ch2] *= 1.0 
        grad_bias = grad_input[0].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_bias, grad_input[1], grad_weight)
    

    def hook_f3(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[:, ch2:] *= ddecay
        grad_weight[:, 0:ch2] *= 1.0 
        #grad_bias[0:ch] = grad_input[0][0:ch]
        grad_bias = grad_input[0].clone()
    
        return (grad_bias, grad_input[1], grad_weight)
    
    def hook_bn(module, grad_input, grad_output):
        #print(module.running_mean, module.running_var, ">>>>>")
        ch = int(grad_input[2].size(0)*ratio)+1
        grad_weight = grad_input[2].clone()
        grad_weight[ch:] *= ddecay
        grad_bias = grad_input[1].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_input[0], grad_bias, grad_weight)
    
    
    ratio = my_ratio[0]
    net.conv1.register_backward_hook(hook_f0)
    net.bn1.register_backward_hook(hook_bn)
    id_num = 1

    for ii in range(9):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer1[ii].conv1.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer1[ii].conv2.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)

    for ii in range(9):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer2[ii].conv1.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer2[ii].conv2.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)
    
    for ii in range(9):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer3[ii].conv1.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer3[ii].conv2.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)

    net.fc.register_backward_hook(hook_f3)


def hook_layers_resnet110(net, my_ratio=0.8):
    ratio = my_ratio[0]

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = grad_input[1].clone()
        grad_weight[:ch, :, :, :] *= 1.0 
        grad_weight[ch:, :, :, :] *= ddecay
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None

        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio2)
        grad_weight = grad_input[1].clone()
        grad_weight[ch:, :, :, :] *= ddecay
        grad_weight[:, ch2:, :, :] *= ddecay
        grad_weight[0:ch, 0:ch2, :, :] *= 1.0
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[ch:, :] *= ddecay
        grad_weight[:, ch2:] *= ddecay
        grad_weight[0:ch, 0:ch2] *= 1.0 
        grad_bias = grad_input[0].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_bias, grad_input[1], grad_weight)
    

    def hook_f3(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[:, ch2:] *= ddecay
        grad_weight[:, 0:ch2] *= 1.0 
        #grad_bias[0:ch] = grad_input[0][0:ch]
        grad_bias = grad_input[0].clone()
    
        return (grad_bias, grad_input[1], grad_weight)

    def hook_bn(module, grad_input, grad_output):
        #print(module.running_mean, module.running_var, ">>>>>")
        ch = int(grad_input[2].size(0)*ratio)+1
        grad_weight = grad_input[2].clone()
        grad_weight[ch:] *= ddecay
        grad_bias = grad_input[1].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_input[0], grad_bias, grad_weight)
    
    '''
    for iii in range(100):
        if isinstance(net.features[iii], nn.Conv2d):
            print(iii, "PPPPPPPPPPPPPPPP")
    '''
    ratio = my_ratio[0]
    net.conv1.register_backward_hook(hook_f0)
    net.bn1.register_backward_hook(hook_bn)
    id_num = 1

    for ii in range(18):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer1[ii].conv1.register_backward_hook(hook_f1)
        net.layer1[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer1[ii].conv2.register_backward_hook(hook_f1)
        net.layer1[ii].bn2.register_backward_hook(hook_bn)

    for ii in range(18):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer2[ii].conv1.register_backward_hook(hook_f1)
        net.layer2[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer2[ii].conv2.register_backward_hook(hook_f1)
        net.layer2[ii].bn2.register_backward_hook(hook_bn)
    
    for ii in range(18):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer3[ii].conv1.register_backward_hook(hook_f1)
        net.layer3[ii].bn1.register_backward_hook(hook_bn)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.layer3[ii].conv2.register_backward_hook(hook_f1)
        net.layer3[ii].bn2.register_backward_hook(hook_bn)

    net.linear.register_backward_hook(hook_f3)

def hook_layers_resnet50(net, my_ratio=0.8):
    ratio = my_ratio[0]

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = grad_input[1].clone()
        grad_weight[:ch, :, :, :] *= 1.0 
        grad_weight[ch:, :, :, :] *= ddecay
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None

        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio2)
        grad_weight = grad_input[1].clone()
        grad_weight[ch:, :, :, :] *= ddecay
        grad_weight[:, ch2:, :, :] *= ddecay
        grad_weight[0:ch, 0:ch2, :, :] *= 1.0
        if grad_input[2] is not None:
            grad_bias = grad_input[2].clone()
            grad_bias[ch:] *= ddecay
            grad_bias[0:ch] *= 1.0 
        else:
            grad_bias = None
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[ch:, :] *= ddecay
        grad_weight[:, ch2:] *= ddecay
        grad_weight[0:ch, 0:ch2] *= 1.0 
        grad_bias = grad_input[0].clone()
        grad_bias[ch:] *= ddecay
        grad_bias[0:ch] *= 1.0 
    
        return (grad_bias, grad_input[1], grad_weight)
    

    def hook_f3(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio2)
        grad_weight = grad_input[2].clone()
        grad_weight[:, ch2:] *= ddecay
        grad_weight[:, 0:ch2] *= 1.0 
        #grad_bias[0:ch] = grad_input[0][0:ch]
        grad_bias = grad_input[0].clone()
    
        return (grad_bias, grad_input[1], grad_weight)
    
    '''
    for iii in range(100):
        if isinstance(net.features[iii], nn.Conv2d):
            print(iii, "PPPPPPPPPPPPPPPP")
    '''
    ratio = my_ratio[0]
    net.module.conv1.register_backward_hook(hook_f0)
    id_num = 1

    for ii in range(3):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer1[ii].conv1.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer1[ii].conv2.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer1[ii].conv3.register_backward_hook(hook_f1)
        if ii == 0:
            net.module.layer1[ii].downsample[0].register_backward_hook(hook_f1)

    for ii in range(4):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer2[ii].conv1.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer2[ii].conv2.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer2[ii].conv3.register_backward_hook(hook_f1)
        if ii == 0:
            net.module.layer1[ii].downsample[0].register_backward_hook(hook_f1)
    
    for ii in range(6):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer3[ii].conv1.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer3[ii].conv2.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer3[ii].conv2.register_backward_hook(hook_f1)
        if ii == 0:
            net.module.layer1[ii].downsample[0].register_backward_hook(hook_f1)

    for ii in range(3):
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer4[ii].conv1.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer4[ii].conv2.register_backward_hook(hook_f1)
        ratio = my_ratio[id_num]
        ratio2 = my_ratio[id_num-1]
        id_num += 1
        net.module.layer4[ii].conv2.register_backward_hook(hook_f1)
        if ii == 0:
            net.module.layer1[ii].downsample[0].register_backward_hook(hook_f1)

    net.module.fc.register_backward_hook(hook_f3)


def train(train_loader, net, criterion, optimizer, epoch, device, cut_ratio=0.8, last_cut=0, hook_flag=False):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))
    loader_length = len(train_loader)

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            epm = np.exp( min( (epoch+batch_index/loader_length) * 3., 10 ) )
            ddecay = torch.pow(mydecay, epm)

            outputs = net(inputs, epoch=epoch-last_cut, batch_ind=batch_index/loader_length)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            epm = np.exp( min( (epoch+batch_index/loader_length) * 5., 15 ) )
            ddecay = torch.pow(mydecay, epm) * 0.0
            #print(ddecay, "JJJJJJJJJ")
            outputs = net(inputs, epoch=epoch-last_cut, batch_ind=batch_index/loader_length, hook_flag=hook_flag)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        #nn.utils.clip_grad_norm(net.parameters(), .1, norm_type=2)
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc

def test(test_loader, net, criterion, optimizer, epoch, device, last_cut=0, hook_flag=False):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, epoch-last_cut, .95, hook_flag, True)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc

cut_ratio = []
total_para = 0
total_flops = 0
tpara = 0
tflops = 0

def norm_pca(net):
    global total_para
    global total_flops
    global tpara
    global tflops
    pca = PCA(.98)
    for name1, item1 in net.named_parameters():
         if item1.dim() > 1:
             weight = item1.view(item1.size(0), -1)
             weight = weight / torch.norm(weight, p=2, dim=1, keepdim=True)
             weight = weight.detach().cpu().numpy()
             pca.fit(weight)

             total_para += np.prod(item1.size())
             tpara += np.prod(item1.size()) * pca.n_components_ / item1.size(0)
             total_flops += np.prod(item1.size())
             if len(cut_ratio) > 0:
                 tflops += np.prod(item1.size()) * pca.n_components_ / item1.size(0) * cut_ratio[-1]
             else:
                 tflops += np.prod(item1.size()) * pca.n_components_ / item1.size(0)

             cut_ratio.append(pca.n_components_ / item1.size(0))
             print(item1.size(), pca.n_components_, pca.n_components_ / item1.size(0), "PPPPPPPP")

def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)
    
    ##这里不影响最终的裁剪率，因为最后其对应的输入会被减掉，这里这是为了刚开始是能尽量保留较多的梯度信息
    config.rate = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. ]
    config.ratio = list(1.0 - np.ones((56)) * 0.45 )

    #config.rate = list( np.ones((55)) )
    #config.ratio = list(1.0 - np.power(np.array([0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4]), 1.0/1) )
    #config.rate = list( np.ones((109)) )
    #config.ratio = list( 1.0 - np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2]) )
    
    #config.rate = list( np.ones((54)) )
    #config.ratio = list( 1.0 - np.array([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]) )
    # define netowrk
    net = get_model(config)

    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        #net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    #net.to(device)
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            ckpt_file_name = '/home/biolab/Desktop/QSM/PRUN/FilterSketch/vgg_16_bn.pt'

            pruned_checkpoint = torch.load(ckpt_file_name, map_location='cuda:0')
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            if False:
                tmp_ckpt = pruned_checkpoint
            else:
                tmp_ckpt = pruned_checkpoint['state_dict']
            
            unsed = []
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v
                unsed.append(k.replace('module.', ''))
            
            net_state_dict = net.state_dict()
            for k, v in net_state_dict.items():
                if k.replace('module.', '') in unsed:
                    pass
                else:
                    new_state_dict[k.replace('module.', '')] = v
            #net.load_state_dict(pruned_checkpoint)
            net.load_state_dict(new_state_dict)
            '''
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)
            '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    #device_ids=[0,1]

    net = net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=device_ids)

    #norm_pca(net)
    #print(tpara / total_para, tflops / total_flops, "OOOOOOOOOOOOOOOOOOOOOOOO")
    #config.ratio = cut_ratio

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config) 
    '''
    data_tmp = imagenet.Data('/media/biolab/NVME/TRACKING_DATASET/IMAGENET/')
    train_loader = data_tmp.loader_train
    test_loader = data_tmp.loader_test
    '''

    logger.info("            =======  Training  =======\n")
    switch_point = [15]

    last_cut = 0
    hook_flag = True
    #config.ratio = config.rate #(config.epochs - epoch) / config.epochs

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if hook_flag:
        hook_layers_vgg16(net, my_ratio=config.ratio)

    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        '''
        if epoch in switch_point:
            config.rate = np.array(config.rate) * np.array(config.ratio)
            if epoch == max(switch_point):
                config.ratio = list( np.ones((55)) )
                hook_flag = False

            t_net = get_model(config)
            t_net_dict = t_net.state_dict()
            #print(t_net, "<<<<<<<<<")

            cout, cin, ch, cw = -1, -1, -1, -1
            for name1, item1 in net.named_parameters():
                for name2, item2 in t_net.named_parameters():
                    if name1 == name2:
                        if item1.dim() > 2:
                            if cout > 0:
                                t_net_dict[name2] = item1[0:item2.size(0), 0:cout, :, :]
                            else:
                                t_net_dict[name2] = item1[0:item2.size(0), :, :, :]
                            cout, cin, ch, cw = item2.size()

                        elif item1.dim() > 1 and item1.dim() < 3:
                            if 'linear1' in name2:
                                t_net_dict[name2] = item1[0:item2.size(0), 0:cout]
                            else:
                                t_net_dict[name2] = item1[:, 0:item2.size(1)]
                        else:
                            if 'linear2' in name2:
                                t_net_dict[name2] = item1[:]
                            else:
                                t_net_dict[name2] = item1[0:item2.size(0)]      

            t_net.load_state_dict(t_net_dict)
            net = t_net
            
            optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=5e-4)
            
            logger.info(net)
            logger.info(" == total parameters: " + str(count_parameters(net)))
            # CPU or GPU
            device = 'cuda' if config.use_gpu else 'cpu'
            # data parallel for multiple-GPU
            if device == 'cuda':
                cudnn.benchmark = True
            net.to(device)
            
            last_cut = epoch
        '''
        train(train_loader, net, criterion, optimizer, epoch, device, cut_ratio=config.ratio, last_cut=last_cut, hook_flag=hook_flag)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device, last_cut=last_cut, hook_flag=hook_flag)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))
 

if __name__ == "__main__":
    main()


'''
    layer = net.features[0]

    #print(layer)
    layer.register_backward_hook(hook_f)
    
    cout, cin, ch, cw = layer.weight.size()
    
    mask_weight = torch.zeros_like(layer.weight)
    mask_weight[int(cout*ratio):, :, :, :] = 0.0
    mask_bias = torch.zeros_like(layer.bias)
    mask_bias[int(cout*ratio):] = 0.0
    layer.weight.register_hook(hookk)
    layer.bias.register_hook(lambda grad: grad*mask_bias)
    
    layer1 = net.features[3]
    mask_weight1 = torch.zeros_like(layer1.weight)
    mask_weight1[int(layer1.weight.size(0)*ratio):, :, :, :] = 0.0
    mask_weight1[:, int(cout*ratio):, :, :] = 0.0
    cout, cin, ch, cw = layer1.weight.size()
    mask_bias1 = torch.zeros_like(layer1.bias)
    mask_bias1[int(cout*ratio):] = 0.0
    layer1.weight.register_hook(lambda grad: grad * mask_weight1)
    layer1.bias.register_hook(lambda grad: grad*mask_bias1)
    
    layer2 = net.features[6]
    mask_weight2 = torch.ones_like(layer2.weight)
    mask_weight2[int(layer2.weight.size(0)*ratio):, :, :, :] = 0.0
    mask_weight2[:, int(cout*ratio):, :, :] = 0.0
    cout, cin, ch, cw = layer2.weight.size()
    mask_bias2 = torch.ones_like(layer2.bias)
    mask_bias2[int(cout*ratio):] = 0.0
    layer2.weight.register_hook(lambda grad: grad * mask_weight2)
    layer2.bias.register_hook(lambda grad: grad*mask_bias2)

    layer3 = net.features[8]
    mask_weight3 = torch.ones_like(layer3.weight)
    mask_weight3[int(layer3.weight.size(0)*ratio):, :, :, :] = 0.0
    mask_weight3[:, int(cout*ratio):, :, :] = 0.0
    cout, cin, ch, cw = layer3.weight.size()
    mask_bias3 = torch.ones_like(layer3.bias)
    mask_bias3[int(cout*ratio):] = 0.0
    layer3.weight.register_hook(lambda grad: grad * mask_weight3)
    layer3.bias.register_hook(lambda grad: grad*mask_bias3)

    layer4 = net.features[10]
    mask_weight4 = torch.ones_like(layer4.weight)
    mask_weight4[int(layer4.weight.size(0)*ratio):, :, :, :] = 0.0
    mask_weight4[:, int(cout*ratio):, :, :] = 0.0
    cout, cin, ch, cw = layer4.weight.size()
    mask_bias4 = torch.ones_like(layer4.bias)
    mask_bias4[int(cout*ratio):] = 0.0
    layer4.weight.register_hook(lambda grad: grad * mask_weight4)
    layer4.bias.register_hook(lambda grad: grad*mask_bias4)
'''

'''
def hook_layers(net, ratio=0.8):

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_weight[:, ch2:, :, :] = 0.0
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[2])
        grad_weight[:, ch2:] = 0.0
        grad_bias = torch.zeros_like(grad_input[0])
        grad_bias[0:ch] = grad_input[0][0:ch]
    
        return (grad_bias, grad_input[1], grad_weight)
        
    layer0 = net.features[0]
    layer0.register_backward_hook(hook_f0)
    
    layer1 = net.features[3]
    layer1.register_backward_hook(hook_f1)

    layer2 = net.features[6]
    layer2.register_backward_hook(hook_f1)

    layer3 = net.features[8]
    layer3.register_backward_hook(hook_f1)

    layer4 = net.features[10]
    layer4.register_backward_hook(hook_f1)

    fc_layer = net.fc
    fc_layer.register_backward_hook(hook_f2)

def hook_layers_vgg19(net, ratio=0.8):

    def hook_f0(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f1(module, grad_input, grad_output):
        ch = int(grad_input[1].size(0)*ratio)
        ch2 = int(grad_input[1].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[1])
        grad_weight[0:ch, :, :, :] = grad_input[1][0:ch, :, :, :]
        grad_weight[:, ch2:, :, :] = 0.0
        grad_bias = torch.zeros_like(grad_input[2])
        grad_bias[0:ch] = grad_input[2][0:ch]
    
        return (grad_input[0], grad_weight, grad_bias)

    def hook_f2(module, grad_input, grad_output):
        ch = int(grad_input[2].size(0)*ratio)
        ch2 = int(grad_input[2].size(1)*ratio)
        grad_weight = torch.zeros_like(grad_input[2])
        grad_weight[:, ch2:] = 0.0
        grad_bias = torch.zeros_like(grad_input[0])
        grad_bias[0:ch] = grad_input[0][0:ch]
    
        return (grad_bias, grad_input[1], grad_weight)
    
    con_list = [3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]

    net.features[0].register_backward_hook(hook_f0)
    for ii in con_list:
        net.features[ii].register_backward_hook(hook_f1)

    fc_layer = net.classifier
    fc_layer.register_backward_hook(hook_f2)
'''