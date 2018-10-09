import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from consts import *
from utils import *
from Leapfrog import Leapfrog
import time

class ResNetFc(nn.Module):
    def __init__(self, num_layers, antisymmetric=False, leapfrog=False, dropOut=False, l2=False, h=0.1):
        self.h = h
        self.num_layers = num_layers
        super(ResNetFc, self).__init__()
        self.fcs = nn.Linear(IMAGE_DIMENSIONS*CHANNELS, WIDTH)
        self.fcf = nn.Linear(WIDTH, NUM_CLASSES)
        self.fc = nn.ModuleList()
        self.leapfrog = leapfrog
        self.dropOut = dropOut
        ### Temporarit constraint, also l2 is not implemented
        # if antisymmetric + leapfrog + dropOut + l2 > 1:
        #     raise Exception("Use only one regulariser")

        for i in range(self.num_layers):
            if antisymmetric:
                self.fc.append(AntiSymmetric())
            elif leapfrog:
                self.fc.append(Leapfrog())
            else:
                self.fc.append(nn.Linear(WIDTH, WIDTH))
        if dropOut:
            self.do = nn.ModuleList()
            for i in range(self.num_layers):
                self.do.append(nn.Dropout(0.7))
        self.fc.apply(init_weights)
    def forward(self, x):
        x = x.view(-1, IMAGE_DIMENSIONS*CHANNELS)
        x = F.relu(self.fcs(x))
        if self.leapfrog:
            h = self.h
            prev = x
            if self.num_layers > 0:
                x = x + h*h*F.relu(self.fc[0](x))
            for i in range(1, self.num_layers):
                temp = x
                x = 2*x - prev + h*h*F.relu(self.fc[i](x))
                prev = temp
        else:
            for i in range(self.num_layers):
                # print('Layer ' + str(i) + ': ' + str(torch.norm(self.fc[i].weight)))
                # print('Layer ' + str(i) + ': ' + str(torch.norm(self.fc[i].bias)))
                if self.dropOut:
                    x = x + F.relu(self.do[i](self.fc[i](x)))
                else:
                    x = x + F.relu(self.fc[i](x))
        # time.sleep(1)
        x = self.fcf(x)
        return x