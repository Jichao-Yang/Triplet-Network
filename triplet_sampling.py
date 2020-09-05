#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
from progress.bar import Bar
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from lib.architecture import Network
from lib.loss import ContrastiveLoss
import lib.toolbox as tb

torch.cuda.manual_seed_all(100100)
torch.manual_seed(100100)
np.random.seed(100100)

#%%
#Loads training and testing datasets

cifar10_train = dsets.CIFAR10("./database", download=True, train=True)
cifar10_test = dsets.CIFAR10("./database", download=True, train=False)

x_train, y_train, x_test, y_test = tb.unpack(cifar10_train, cifar10_test)
train_size = len(x_train)
test_size = len(x_test)

#%%
#Define Model Structure

net = Network()
net.cuda()

#%%
#Initializes training

epoch = 200
batch_size = 100
contrastiveLoss = ContrastiveLoss(1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
lowest_loss = 0
num_class = 10
num_train_instance = 1500
num_test_instance = 300

#%%
#Training the network
#Note that we are only encoding the images, not classifying at this point.

for epoch in range(1,epoch+1):
    x0_train,x1_train,tgt_train = tb.random_pair(x_train,y_train,num_train_instance)
    x0_test,x1_test,tgt_test = tb.random_pair(x_test,y_test,num_test_instance)

    net.train()
    bar1 = Bar('train\t\t', max=int(num_train_instance/batch_size))
    for i in range(int(num_train_instance/batch_size)):
        #trains network
        x0_batch = torch.from_numpy(x0_train[i*batch_size:(i+1)*batch_size]).cuda()         # pylint: disable=no-member # pytorch/issues/701
        x1_batch = torch.from_numpy(x1_train[i*batch_size:(i+1)*batch_size]).cuda()         # pylint: disable=no-member # pytorch/issues/701
        tgt_batch = torch.from_numpy(tgt_train[i*batch_size:(i+1)*batch_size]).cuda()         # pylint: disable=no-member # pytorch/issues/701

        optimizer.zero_grad()
        output1, output2 = net(x0_batch, x1_batch)
        loss = contrastiveLoss(output1, output2, tgt_batch)
        loss.backward()
        optimizer.step()

        bar1.next()
    bar1.finish()

    net.eval()
    with torch.no_grad():
        #Loss on train set
        loss2 = 0
        bar2 = Bar('Train eval\t', max=int(num_train_instance/batch_size))
        for i in range(int(num_train_instance/batch_size)):
            x0_batch = torch.from_numpy(x0_train[i*batch_size:(i+1)*batch_size]).cuda()     # pylint: disable=no-member # pytorch/issues/701
            x1_batch = torch.from_numpy(x1_train[i*batch_size:(i+1)*batch_size]).cuda()     # pylint: disable=no-member # pytorch/issues/701
            tgt_batch = torch.from_numpy(tgt_train[i*batch_size:(i+1)*batch_size]).cuda()     # pylint: disable=no-member # pytorch/issues/701
            
            output1, output2 = net(x0_batch, x1_batch)
            loss = contrastiveLoss(output1, output2, tgt_batch, size_average=False)
            loss2 += float(loss)
            bar2.next()
        bar2.finish()
        train_loss.append(loss2/num_train_instance)

        #Loss on test set
        loss3 = 0
        bar3 = Bar('Test eval\t', max=int(num_test_instance/batch_size))
        for i in range(int(num_test_instance/batch_size)):
            x0_batch = torch.from_numpy(x0_test[i*batch_size:(i+1)*batch_size]).cuda() # pylint: disable=no-member # pytorch/issues/701
            x1_batch = torch.from_numpy(x1_test[i*batch_size:(i+1)*batch_size]).cuda() # pylint: disable=no-member # pytorch/issues/701
            tgt_batch = torch.from_numpy(tgt_test[i*batch_size:(i+1)*batch_size]).cuda() # pylint: disable=no-member # pytorch/issues/701

            output1, output2 = net(x0_batch, x1_batch)
            loss = contrastiveLoss(output1, output2, tgt_batch, size_average=False)
            loss3 += float(loss)
            bar3.next()
        bar3.finish()
        test_loss.append(loss3/num_test_instance)

    #Save Network parameters
    if epoch == 1:
        lowest_loss = test_loss[0]
        torch.save(net, 'best_param.pt')
    elif test_loss[-1] < lowest_loss:
        torch.save(net, 'best_param.pt')
    
    #Evaluate current network using KNN
    torch.save(net, 'last_param.pt')
    encoded_test, encoded_train = tb.encode(x_train,x_test,128,'last_param.pt')
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn.fit(encoded_train, y_train)
    train_acc.append(knn.score(encoded_train,y_train))
    knn.fit(encoded_test, y_test)
    test_acc.append(knn.score(encoded_test,y_test))


    print('\repoch =', epoch)
    print('\rtrain loss =', round(train_loss[-1],5), '\ttest loss =', round(test_loss[-1],5))
    print('\rtrain acc  =', round(train_acc[-1],5), '\ttest acc  =', round(test_acc[-1],5))

    #Plot all values
    plt.plot(range(1,epoch+1), train_loss, label='train loss')
    plt.plot(range(1,epoch+1), test_loss, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    plt.plot(range(1,epoch+1), train_acc, label='train acc')
    plt.plot(range(1,epoch+1), test_acc, label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('acc.png')
    plt.close()

#%%
#Encode test images
print('\ttraining finished')
encoded_test, encoded_train = tb.encode(x_train, x_test, 128, 'best_param.pt', to_csv=True)

#%%
#Visualize encoded vectors
pca = PCA(n_components=2)
pca.fit(encoded_train)

reduced_train = pca.transform(encoded_train)
reduced_test = pca.transform(encoded_test)

color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for i in range(num_class):
    plt.scatter([reduced_train[:,0][j] for j in np.where(y_train==i)][0],
                [reduced_train[:,1][j] for j in np.where(y_train==i)][0],
                s=0.5,
                c=color[i])
plt.savefig('Encoded_train_set.png', dpi=300)
plt.close()

for i in range(num_class):
    plt.scatter([reduced_test[:,0][j] for j in np.where(y_test==i)][0],
                [reduced_test[:,1][j] for j in np.where(y_test==i)][0],
                s=0.5,
                c=color[i])
plt.savefig('Encoded_test_set.png', dpi=300)
plt.close()
