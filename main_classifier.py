import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F

print('torch version is {}'.format(torch.__version__))
if torch.cuda.is_available():
	print('cuda is available')
else:
	print('cuda is not avaibalbe')

torch.cuda.manual_seed_all(100100)
torch.manual_seed(100100)
np.random.seed(100100)




cifar10_train = dsets.CIFAR10("./database", download=True, train=True)
cifar10_test = dsets.CIFAR10("./database", download=True, train=False)

datasize = len(cifar10_train)
datasize_test = len(cifar10_test)
x_train = []
x_test = []
y_train = []
y_test = []

#Unpacks image to array
for i in range(datasize):
	y_train.append(cifar10_train[i][1])
for i in range(datasize_test):
	y_test.append(cifar10_test[i][1])

x_train = np.loadtxt('encoded_train_set.csv', delimiter=',')
x_test = np.loadtxt('encoded_test_set.csv', delimiter=',')

y_train = np.array(y_train)
y_test = np.array(y_test)

print('Dataset downloaded')




class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc = nn.Linear(128, 10)

	def forward(self, x):
		x = self.fc(x)
		return x

net = Network()
net.cuda()

softmax_cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)


train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]
epoch=100
batchsize=100

for epoch in range(1, epoch+1):
	print('epoch', epoch)
	perm = np.random.permutation(datasize)
	for i in range(0, datasize, batchsize):
		#Create mini-batch.
		x_batch = x_train[perm[i:i+batchsize]]
		y_batch = y_train[perm[i:i+batchsize]]
	
		#Convert a numpy array to a tensor to do gradient calculations.
		if torch.cuda.is_available():
			x_batch = torch.from_numpy(x_batch).float().cuda()
			y_batch = torch.from_numpy(y_batch).long().cuda()
		else:
			x_batch = torch.from_numpy(x_batch).float()
			y_batch = torch.from_numpy(y_batch).long()

		#Initialize the stored gradient.
		optimizer.zero_grad()

		#Get the model output for the input mini-batch.
		y = net(x_batch)
	
		#Calculate loss from the output of the model. At this time, the activation function of the output layer is also reflected in the calculation.
		loss = softmax_cross_entropy(y, y_batch)  
	
		#From loss, calculate the gradient of each parameter.
		loss.backward()

		#Each parameter is updated from the calculated gradient.
		optimizer.step()



	sum_score = 0
	sum_loss = 0
  
	#Evaluate the model with train data.
	for i in range(0, datasize, batchsize):
		x_batch = x_train[i:i+batchsize]
		y_batch = y_train[i:i+batchsize]
		if torch.cuda.is_available():
			x_batch = torch.from_numpy(x_batch).float().cuda()
			y_batch = torch.from_numpy(y_batch).long().cuda()
		else:
			x_batch = torch.from_numpy(x_batch).float()
			y_batch = torch.from_numpy(y_batch).long()
		y = net(x_batch)
		loss = softmax_cross_entropy(y, y_batch)
		sum_loss += float(loss.cpu().data.item()) * batchsize
		_, predict = y.max(1)
		sum_score += predict.eq(y_batch).sum().item()
	print("\ttrain  mean loss={}, accuracy={}".format(sum_loss / datasize, sum_score / datasize))
	train_loss.append(sum_loss / datasize)
	train_acc.append(sum_score / datasize)


	sum_score = 0
	sum_loss = 0

  
	#Evaluate the model with test data.
	for i in range(0, datasize_test, batchsize):
		x_batch = x_test[i:i+batchsize]
		y_batch = y_test[i:i+batchsize]
		if torch.cuda.is_available():
			x_batch = torch.from_numpy(x_batch).float().cuda()
			y_batch = torch.from_numpy(y_batch).long().cuda()
		else:
			x_batch = torch.from_numpy(x_batch).float()
			y_batch = torch.from_numpy(y_batch).long()
		y = net(x_batch)
		loss = softmax_cross_entropy(y, y_batch)
		sum_loss += float(loss.cpu().data.item()) * batchsize
		_, predict = y.max(1)
		sum_score += predict.eq(y_batch).sum().item()
	print("\ttest  mean loss={}, accuracy={}".format(sum_loss / datasize_test, sum_score / datasize_test))
	test_loss.append(sum_loss / datasize_test)
	test_acc.append(sum_score / datasize_test)


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
	plt.ylabel('accuracy')
	plt.legend()
	plt.savefig('acc.png')
	plt.close()