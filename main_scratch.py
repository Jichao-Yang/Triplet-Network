import toolbox as tb
import torchvision.datasets as dsets
train = dsets.CIFAR10("./database", download=True, train=True)
test = dsets.CIFAR10("./database", download=True, train=False)
a,b,c,d = tb.unpack(train,test)
pass