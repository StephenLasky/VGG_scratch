import torch
import torch.nn as nn
import torchvision

import random
import math



# create training data
print "Started"
class DataPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        return str(self.x) + " -> " + str(self.y)
data = []
for i in range(0,1000):
    a = float(random.randint(0,100))
    b = float(random.randint(0,100))
    x = torch.tensor([a,b])
    y = torch.tensor([a+b])
    data.append(DataPoint(x,y))



class AdderNet(nn.Module):
    def __init__(self, num_hidden, hidden_width):
        super(AdderNet, self).__init__()
        self.relu = nn.ReLU()

        self.hiddenLayers = []
        self.inputLayer = nn.Linear(2, hidden_width)
        self.outputLayer = nn.Linear(hidden_width, 1)

        for i in range(num_hidden):
            self.hiddenLayers.append(nn.Linear(hidden_width, hidden_width))

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.relu(out)

        for layer in self.hiddenLayers:
            out = layer(out)
            out = self.relu(out)

        out = self.outputLayer(out)

        return out

hidden_width = 512
num_hidden = 8
num_epochs = 10
learning_rate = 0.001

model = AdderNet(num_hidden, hidden_width)
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print model.forward(torch.Tensor([2,2]))

for epoch in range(num_epochs):
    avgLoss = torch.tensor([0.0]).item()
    for i in range(0,len(data)):
        d = data[i]
        x = d.x
        y = d.y

        out = model.forward(x)
        loss = lossFunction(out, y)
        avgLoss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print avgLoss / len(data)



# test
avgLoss = 0
test_size = 50
for i in range(0,test_size):
    a = float(random.randint(0,100))
    b = float(random.randint(0,100))
    x = torch.tensor([a,b])
    y = torch.tensor([a+b])
    out = model(x)
    loss = abs(out.item() - y.item())
    avgLoss += loss

avgLoss /= test_size
print "test loss:", avgLoss
