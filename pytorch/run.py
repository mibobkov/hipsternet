import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ResNet import ResNet
import ResNetFc
import AntiSymmetric
from Multilevel import Multilevel
from utils import *
from train import train_net
from consts import *
import argparse

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

DATASET_DOWNLOAD_FUNC = torchvision.datasets.CIFAR10

trainset = DATASET_DOWNLOAD_FUNC(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = DATASET_DOWNLOAD_FUNC(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', default='2', type=int)
    parser.add_argument('--net', default='resnet', choices='resnet, resnetfc')
    parser.add_argument('--opt', default='none', choices=('none', 'anti'))
    parser.add_argument('--multi', defaul='none', choices=('none', 'random', 'copy', 'interleave', 'interpolate'))
    ns = parser.parse_args(sys.argv[1:])
    if ns.net == 'resnet':
        if ns.multi == 'none':
            method=None
        elif ns.multi == 'random':
            method=Multilevel.random
        elif ns.multi == 'copy':
            method=Multilevel.copy
        elif ns.multi == 'interleave':
            method=Multilevel.interleave
        elif ns.multi == 'interpolate':
            method=Multilevel.interpolate

        train_net(ResNet(ns.layers), testloader, trainloader, method)
    elif ns.net == 'resnetfc':
        if ns.opt == 'anti':
            train_net(ResNetFc(ns.layers, antisymmetric=True))
        else:
            train_net(ResNetFc(ns.layers, antisymmetric=False))
