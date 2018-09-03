import time
import torch
import torch.optim as optim
from utils import *
from consts import *

def train_net(net, testloader, trainloader, method=None):
    # print('1')
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#, weight_decay=0.005)
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
    for epoch in range(40):  # loop over the dataset multiple times
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
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            totaltime += time.time()-start
            # print('6')
            if i % 1000 == 999:    # print every 2000 mini-batches
                accuracy = getAccuracy(net, testloader)
                print('E:%d, iter-%5d loss: %.3f, time: %.3f, validation: %.3f %%, train acc.: %.3f %%' %
                      (epoch + 1, i+1,  running_loss / 2000, totaltime, 100 * accuracy, train_correct/train_total*100))
                f.write('E:%d, iter-%d loss: %.3f, time: %.3f, validation: %.3f %%, train acc.: %.3f %%\n' %
                      (epoch + 1, i+1,  running_loss / 2000, totaltime, 100 * accuracy, train_correct/train_total*100))
                train_correct = 0
                train_total = 0
                running_loss = 0.0
            i+=1
            
            if i % 2000 == 1999 and method != None:
                net.doubleLayers(method)
                net.to(device)
    f.close()