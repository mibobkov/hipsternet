import torch
import torch.nn as nn
from AntiSymmetric import AntiSymmetric
from consts import device
def getAccuracy(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=0.01)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=0.01)
        m.bias.data.fill_(0.01)
    if type(m) == AntiSymmetric:
        torch.nn.init.xavier_normal_(m.weight, gain=0.01)