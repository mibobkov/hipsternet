import time
import torch
import torch.optim as optim
import numpy as np
from utils import *
from consts import *
from torch.autograd import Variable

def train_net(net, testloader, trainloader, method=None, h=0.1, double=1000):
    # vis = visdom.Visdom()
    # print('1')
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.04)#, weight_decay=0.005)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if method == None:
        methodstring =  ''
    else:
        methodstring = str(method)
    # print('2')
    f= open("logs/" + str(net.num_layers)+methodstring+timestr,"w+", 1)
    print(net.num_layers)
    f.write('Initial number of layers: %d\n' % (net.num_layers))
    if method != None:
        f.write('Method: ' + str(method))
    #optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    totaltime = 0
    i = 0
    # print('3')
    for epoch in range(80):  # loop over the dataset multiple times
        # a = net.fcs.weight.data.cpu().numpy()[0].reshape(28, 28)
        # b = (a - np.min(a)) / np.ptp(a)
        # vis.image(b)
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        # print('4')
        for j, data in enumerate(trainloader, 0):
            # print('5')
            start = time.time()
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted==labels).sum().item()
            loss = criterion(outputs, labels)
            # regloss = torch.tensor(0.0, device=device)
            # for k in range(0, net.num_layers-1):
            #     regloss += 2 * torch.norm(net.fc[k + 1].weight - net.fc[k].weight)
            #     regloss += torch.norm(net.fc[k + 1].bias - net.fc[k].bias)
            # loss = loss + regloss/h
            loss.backward()
            # newloss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            totaltime += time.time()-start
            # print('6')
            i+=1

            if i % double == double-1 and method != None:
                net.doubleLayers(method)
                net.to(device)
        if epoch % 2 == 0:
            net.eval()
            accuracy = getAccuracy(net, testloader)
            net.train()
            print('E:%d, iter-%5d loss: %.3f, time: %.3f, validation: %.3f %%, train acc.: %.3f %%' %
                  (epoch + 1, i+1,  running_loss / 2000, totaltime, 100 * accuracy, train_correct/train_total*100))
            f.write('E:%d, iter-%d loss: %.3f, time: %.3f, validation: %.3f %%, train acc.: %.3f %%\n' %
                    (epoch + 1, i+1,  running_loss / 2000, totaltime, 100 * accuracy, train_correct/train_total*100))
    f.close()