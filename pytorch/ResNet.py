import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from consts import *
from utils import *
from Multilevel import Multilevel

class ResNet(nn.Module):
    def __init__(self, num_layers):
        self.num_layers = num_layers
        super(ResNet, self).__init__()
        self.fcf = nn.Linear(CONV_WIDTH*IMAGE_DIMENSIONS, NUM_CLASSES)
        self.convs = nn.Conv2d(CHANNELS, CONV_WIDTH, 5, padding=(2, 2))
        self.conv = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv.append(nn.Conv2d(CONV_WIDTH, CONV_WIDTH, KERNEL_SIZE, padding=(2, 2)))
                #AntiSymmetric())
        self.conv.apply(init_weights)
    
    def forward(self, x):
        x = F.relu(self.convs(x))
        for i in range(self.num_layers):
            x = x + F.relu(self.conv[i](x))
        x = x.view(-1, CONV_WIDTH*IMAGE_DIMENSIONS)
        x = F.relu(self.fcf(x))
        return x
    
    def doubleLayers(self, method):
        if self.num_layers == 8:
            return
        for i in range(0, self.num_layers):
            self.conv.append(nn.Conv2d(CONV_WIDTH, CONV_WIDTH, KERNEL_SIZE, padding=(2, 2)))
        if method == Multilevel.random:
            self.conv[self.num_layers:self.num_layers*2].apply(init_weights)
        elif method == Multilevel.copy:
            for i in range(0, self.num_layers):
                self.conv[i].weight.data = self.conv[i].weight.data/2
                self.conv[i].bias.data = self.conv[i].bias.data/2
                self.conv[i+self.num_layers].weight.data = self.conv[i].weight.data.clone()
                self.conv[i+self.num_layers].bias.data = self.conv[i].bias.data.clone()
        elif method == Multilevel.interleave:
            for i in range(0, self.num_layers-1):
                self.conv[self.num_layers*2-2-2*i].weight.data = self.conv[self.num_layers-1-i].weight.data.clone()
                self.conv[self.num_layers*2-2-2*i].bias.data = self.conv[self.num_layers-1-i].bias.data.clone()
            for i in range(0, self.num_layers):
                self.conv[i*2].weight.data = self.conv[i*2].weight.data/2
                self.conv[i*2].bias.data = self.conv[i*2].bias.data/2
                self.conv[i*2+1].weight.data = self.conv[i*2].weight.data.clone()
                self.conv[i*2+1].bias.data = self.conv[i*2].bias.data.clone()
        elif method == Multilevel.interpolate:
            for i in range(0, self.num_layers-1):
                self.conv[self.num_layers*2-2-2*i].weight.data = self.conv[self.num_layers-1-i].weight.data.clone()
                self.conv[self.num_layers*2-2-2*i].bias.data = self.conv[self.num_layers-1-i].bias.data.clone()
            for i in range(0, self.num_layers):
                self.conv[i*2].weight.data = self.conv[i*2].weight.data/2
                self.conv[i*2].bias.data = self.conv[i*2].bias.data/2
                if i == self.num_layers-1:
                    self.conv[i*2+1].weight.data = self.conv[i*2].weight.data.clone()
                    self.conv[i*2+1].bias.data = self.conv[i*2].bias.data.clone()
                else:
                    self.conv[i*2+1].weight.data = (self.conv[i*2].weight.data + self.conv[i*2+2].weight.data)/2
                    self.conv[i*2+1].bias.data = (self.conv[i*2].bias.data + self.conv[i*2+2].bias.data)/2
        self.num_layers = self.num_layers*2