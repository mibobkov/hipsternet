import numpy as np
import hipsternet.input_data as input_data
import hipsternet.neuralnet as nn
from hipsternet.solver import *
import argparse
import sys


n_iter = 15000
alpha = 1e-4
mb_size = 64
n_experiment = 4
reg = 1e-5
print_after = 100
p_dropout = 0.8
loss = 'cross_ent'
nonlin = 'relu'
solver = 'sgd'


def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean


if __name__ == '__main__':
    valid_nets = ('ff', 'cnn', 'resnet')
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='ff', choices = valid_nets)
    parser.add_argument('--opt', default='none', choices = ('none', 'verlet', 'leapfrog', 'antisymmetric'))
    parser.add_argument('--n_iter', default=n_iter, type=int)
    parser.add_argument('--solver', default='sgd', choices=('sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'))
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--step', default = alpha, type=float)
    parser.add_argument('--H', default=256, type=int)
    ns = parser.parse_args(sys.argv[1:])
    net_type = ns.net
    optimisation = ns.opt
    n_iter = ns.n_iter
    solver = ns.solver
    alpha = ns.step
        # if net_type not in valid_nets:
        #     raise Exception('Valid network type are {}'.format(valid_nets))

    print('Net: ' + str(net_type))
    print('Iterations: ' + str(n_iter))
    print('Step size: ' + str(alpha))    
    print('Minibatch size: ' + str(mb_size))
    print('Number of experiments: ' + str(n_experiment))
    print('Nonlinearity: ' + str(nonlin))
    print('Solver: ' + str(solver))
    print('Optimisation: ' + str(optimisation))
    print('Number of layers: ' + str(ns.num_layers))

    mnist = input_data.read_data_sets('data/MNIST_data/', one_hot=False)
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_test, y_test = mnist.test.images, mnist.test.labels

    M, D, C = X_train.shape[0], X_train.shape[1], y_train.max() + 1

    X_train, X_val, X_test = prepro(X_train, X_val, X_test)

    if net_type == 'cnn':
        img_shape = (1, 28, 28)
        X_train = X_train.reshape(-1, *img_shape)
        X_val = X_val.reshape(-1, *img_shape)
        X_test = X_test.reshape(-1, *img_shape)

    solvers = dict(
        sgd=sgd,
        momentum=momentum,
        nesterov=nesterov,
        adagrad=adagrad,
        rmsprop=rmsprop,
        adam=adam
    )

    solver_fun = solvers[solver]
    accs = np.zeros(n_experiment)

    print()
    print('Experimenting on {}'.format(solver))
    print()

    for k in range(n_experiment):
        print('Experiment-{}'.format(k + 1))

        # Reset model
        if net_type == 'ff':
            net = nn.FeedForwardNet(D, C, H=128, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin)
        elif net_type == 'cnn':
            net = nn.ConvNet(10, C, H=128)
        elif net_type == 'resnet':
            net = nn.ResNet(D, C, H=ns.H, lam=reg, p_dropout=p_dropout, loss=loss, nonlin=nonlin, optimisation=optimisation, num_layers=ns.num_layers)

        net = solver_fun(
            net, X_train, y_train, val_set=(X_val, y_val), mb_size=mb_size, alpha=alpha,
            n_iter=n_iter, print_after=print_after
        )

        y_pred = net.predict(X_test)
        accs[k] = np.mean(y_pred == y_test)

    print()
    print('Mean accuracy: {:.4f}, std: {:.4f}'.format(accs.mean(), accs.std()))
