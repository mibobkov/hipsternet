import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from consts import *
from utils import *
from Leapfrog import Leapfrog

class ResNetFc(nn.Module):
    def __init__(self, num_layers, antisymmetric=False, leapfrog=False):
        self.num_layers = num_layers
        super(ResNetFc, self).__init__()
        self.fcs = nn.Linear(IMAGE_DIMENSIONS*CHANNELS, WIDTH)
        self.fcf = nn.Linear(WIDTH, NUM_CLASSES)
        self.fc = nn.ModuleList()
        self.leapfrog = leapfrog

        for i in range(self.num_layers):
            if antisymmetric:
                self.fc.append(AntiSymmetric())
            elif leapfrog:
                self.fc.append(Leapfrog())
            else:
                self.fc.append(nn.Linear(WIDTH, WIDTH))
        self.fc.apply(init_weights)
    def forward(self, x):
        x = x.view(-1, IMAGE_DIMENSIONS*CHANNELS)
        x = F.relu(self.fcs(x))
        if self.leapfrog:
            h = 1
            prev = x
            if self.num_layers > 0:
                x = 2*x + h*h*F.relu(self.fc[0](x))
            for i in range(1, self.num_layers):
                temp = x
                x = 2*x - prev + h*h*F.relu(self.fc[i](x))
                prev = temp
        else:
            for i in range(self.num_layers):
                x = x + F.relu(self.fc[i](x))
        x = F.relu(self.fcf(x))
        return x