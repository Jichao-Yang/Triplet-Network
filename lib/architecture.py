import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256,128)
        self.pool = nn.MaxPool2d(2)
        self.pr = nn.PReLU()

    def forward(self, x0, x1, x2):
        #label_x1 == label_x0

        x0 = self.pool(self.pr(self.conv1(x0)))
        x0 = self.pool(self.pr(self.conv2(x0)))
        x0 = x0.reshape(-1, 64*5*5)
        x0 = self.pr(self.fc1(x0))
        x0 = self.fc2(x0)

        x1 = self.pool(self.pr(self.conv1(x1)))
        x1 = self.pool(self.pr(self.conv2(x1)))
        x1 = x1.reshape(-1, 64*5*5)
        x1 = self.pr(self.fc1(x1))
        x1 = self.fc2(x1)

        x2 = self.pool(self.pr(self.conv1(x2)))
        x2 = self.pool(self.pr(self.conv2(x2)))
        x2 = x2.reshape(-1, 64*5*5)
        x2 = self.pr(self.fc1(x2))
        x2 = self.fc2(x2)

        return x0,x1,x2

    def encode(self, x1):
        #Takes input 3*32*32, output 128 Dimensional Vector
        x1 = self.pool(self.pr(self.conv1(x1)))
        x1 = self.pool(self.pr(self.conv2(x1)))
        x1 = x1.reshape(-1, 64*5*5)
        x1 = self.pr(self.fc1(x1))
        x1 = self.fc2(x1)
        return x1