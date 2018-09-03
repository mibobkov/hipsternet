import torch
import torch.nn as nn
from consts import *

class Leapfrog(nn.Module):
    def __init__(self):
        super(Leapfrog, self).__init__()
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(WIDTH, WIDTH))
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(WIDTH))
        torch.nn.init.xavier_normal_(self.weight, gain=0.01)
        self.bias.data.fill_(0.01)

    def forward(self, x):
        #w = self.weight
        w = -torch.mm(self.weight.permute(1, 0), self.weight )
        #print(w.size())
        #print(x.size())
        return torch.nn.functional.linear(x, w, self.bias)

