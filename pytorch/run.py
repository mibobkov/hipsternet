import torch
import torchvision
import torchvision.transforms as transforms
from ResNet import ResNet
from ResNetFc import ResNetFc
from Multilevel import Multilevel
from utils import *
from train import train_net
from consts import *
import sys
import argparse


def run():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Change consts when changing dataset
    DATASET_DOWNLOAD_FUNC = torchvision.datasets.CIFAR10

    trainset = DATASET_DOWNLOAD_FUNC(root='./data', train=True,
                                     download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1)

    testset = DATASET_DOWNLOAD_FUNC(root='./data', train=False,
                                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', default='2', type=int)
    parser.add_argument('--net', default='resnet', choices='resnet, resnetfc')
    parser.add_argument('--opt', default='none', choices=('none', 'anti', 'leap'))
    parser.add_argument('--multi', default='none', choices=('none', 'random', 'copy', 'interleave', 'interpolate'))
    ns = parser.parse_args(sys.argv[1:])

    # if ns.net == 'resnet':
    #     if ns.multi == 'none':
    #         method=None
    #     elif ns.multi == 'random':
    #         method=Multilevel.random
    #     elif ns.multi == 'copy':
    #         method=Multilevel.copy
    #     elif ns.multi == 'interleave':
    #         method=Multilevel.interleave
    #     elif ns.multi == 'interpolate':
    #         method=Multilevel.interpolate
    #
    #     train_net(ResNet(ns.layers), testloader, trainloader, method)
    # elif ns.net == 'resnetfc':
    #     if ns.opt == 'anti':
    #         train_net(ResNetFc(ns.layers, antisymmetric=True), testloader, trainloader)
    #     elif ns.opt == 'leap':
    #         train_net(ResNetFc(ns.layers, leapfrog=True), testloader, trainloader)
    #     else:
    #         train_net(ResNetFc(ns.layers, antisymmetric=False), testloader, trainloader)
    # for i in range(0, 1):
    #     train_net(ResNetFc(4), testloader, trainloader)
    # for i in range(0, 3):
        # train_net(ResNetFc(8), testloader, trainloader)
        # train_net(ResNetFc(8, antisymmetric=True), testloader, trainloader)
        # train_net(ResNetFc(8, leapfrog=True, h=1), testloader, trainloader, h=1)
        # train_net(ResNetFc(8, leapfrog=True, h=0.1), testloader, trainloader, h=0.1)
        # train_net(ResNetFc(8, leapfrog=True, h=0.01), testloader, trainloader, h=0.01)
        # train_net(ResNetFc(8, dropOut=True), testloader, trainloader)
    for i in range(0, 5):
        # train_net(ResNet(1), testloader, trainloader, Multilevel.random)
        # train_net(ResNet(1), testloader, trainloader, Multilevel.copy)
        train_net(ResNet(1), testloader, trainloader, Multilevel.interleave)
        # CONV_WIDTH *= 2
        # train_net(ResNet(1), testloader, trainloader, Multilevel.interpolate) #, double=1000)
        # train_net(ResNet(1), testloader, trainloader, Multilevel.interpolate, double=500)
        # train_net(ResNet(1), testloader, trainloader, Multilevel.interpolate, double=2000)
        # train_net(ResNet(1), testloader, trainloader, Multilevel.interpolate, double=3000)

if __name__ == '__main__':
    run()
