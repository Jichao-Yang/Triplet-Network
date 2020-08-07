import numpy as np
import torch
from progress.bar import Bar
import random

from lib.architecture import Network

def unpack(train,test,max_val=255,channel_first=True,normalize=True,dtype='float32'):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(len(train)):
        x_train.append(np.array(train[i][0]))
        y_train.append(train[i][1])
    for i in range(len(test)):
        x_test.append(np.array(test[i][0]))
        y_test.append(test[i][1])
    
    x_train = np.array(x_train).astype(dtype)
    y_train = np.array(y_train).astype(dtype)
    x_test = np.array(x_test).astype(dtype)
    y_test = np.array(y_test).astype(dtype)

    trainshape = x_train.shape
    testshape = x_test.shape
    if channel_first:
        if len(x_train[0].shape) == 2:
            x_train = x_train.reshape((trainshape[0], 1, trainshape[1], trainshape[2]))             # pylint: disable=E1136  # pylint/issues/3139
            x_test = x_test.reshape((testshape[0], 1, testshape[1], testshape[2]))                  # pylint: disable=E1136  # pylint/issues/3139
        elif len(x_train[0].shape) == 3:
            x_train = x_train.reshape((trainshape[0],trainshape[3], trainshape[1], trainshape[2]))  # pylint: disable=E1136  # pylint/issues/3139
            x_test = x_test.reshape((testshape[0],testshape[3], testshape[1], testshape[2]))        # pylint: disable=E1136  # pylint/issues/3139
        else:
            raise IndexError('Dataset dimensions not compatible.')

    if normalize:
        x_train = x_train/max_val
        x_test = x_test/max_val

    return x_train, y_train, x_test, y_test

def encode(x_train, x_test, out_dim, path, batch_size=100, to_csv=True, use_cpu=True):
    #Encodes two image sets
    if use_cpu:
        net = torch.load(path, map_location='cpu')
    else:
        net = torch.load(path)
    net.eval()

    train_size = len(x_train)
    test_size = len(x_test)

    encoded_test = np.empty((test_size, out_dim), dtype='float64')
    encoded_train = np.empty((train_size, out_dim), dtype='float64')

    if use_cpu:
        bar = Bar('Test enc\t', max = int(test_size/batch_size))
        for i in range(0,test_size,batch_size):
            encoded_test[i:i+batch_size] = net.encode(torch.from_numpy(x_test[i:i+batch_size])).detach().numpy()     # pylint: disable=no-member # pytorch/issues/701
            bar.next()
        bar.finish()
        bar = Bar('Train enc\t', max = int(train_size/batch_size))
        for i in range(0,train_size,batch_size):
            encoded_train[i:i+batch_size] = net.encode(torch.from_numpy(x_train[i:i+batch_size])).detach().numpy()   # pylint: disable=no-member # pytorch/issues/701
            bar.next()
        bar.finish()
    else:
        bar = Bar('Test enc\t', max = int(test_size/batch_size))
        for i in range(0,test_size,batch_size):
            encoded_test[i:i+batch_size] = net.encode(torch.from_numpy(x_test[i:i+batch_size]).cuda()).cpu().detach().numpy()     # pylint: disable=no-member # pytorch/issues/701
            bar.next()
        bar.finish()
        bar = Bar('Train enc\t', max = int(train_size/batch_size))
        for i in range(0,train_size,batch_size):
            encoded_train[i:i+batch_size] = net.encode(torch.from_numpy(x_train[i:i+batch_size]).cuda()).cpu().detach().numpy()   # pylint: disable=no-member # pytorch/issues/701
            bar.next()
        bar.finish()

    if to_csv:
        np.savetxt('encoded_test_set.csv', encoded_test, delimiter=',')
        np.savetxt('encoded_train_set.csv', encoded_train, delimiter=',')

    return encoded_test, encoded_train

def instance(x, y, num):
    #x0.label == x1.lable != x2.label
    size = len(y)
    x0=[]; x1=[]; x2=[]
    bar = Bar('Genrt inst\t', max=num)
    for i in range(num):
        index = random.randint(0,size-1)
        x0.append(x[index])
        x1.append(x[np.random.choice(np.where(y == y[index])[0])])
        x2.append(x[np.random.choice(np.where(y != y[index])[0])])
        bar.next()
    bar.finish()
    x0 = np.array(x0); x1 = np.array(x1); x2 = np.array(x2)
    return x0, x1, x2