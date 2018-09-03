import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResNetFc(nn.Module):
    def __init__(self, num_layers, antisymmetric=False):
        self.num_layers = num_layers
        super(ResNetFc, self).__init__()
        self.fcs = nn.Linear(IMAGE_DIMENSIONS*CHANNELS, WIDTH)
        self.fcf = nn.Linear(WIDTH, NUM_CLASSES)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers):
            if antisymmetric:
                self.fc.append(AntiSymmetric())
            else:
                self.fc.append(nn.Linear(WIDTH, WIDTH))
        self.fc.apply(init_weights)
    def forward(self, x):
        x = x.view(-1, IMAGE_DIMENSIONS*CHANNELS)
        x = F.relu(self.fcs(x))
        for i in range(self.num_layers):
            x = x + F.relu(self.fc[i](x))
        x = F.relu(self.fcf(x))
        return x