import torch
import torch.nn as nn
import torchvision

import random
import math
import time

A_MIN = -100
A_MAX = 100
B_MIN = -100
B_MAX = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device   )



# create training data
class DataPoint:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __str__(self):
        return str(self.x) + " -> " + str(self.y)
data = []
for i in range(0,3000):
    a = float(random.randint(A_MIN,A_MAX))
    b = float(random.randint(B_MIN,B_MAX))
    if b == 0:
        b = 1
    x = torch.tensor([a,b], device=device)
    y = torch.tensor([a/b], device=device)
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

        # self.hiddenLayers = nn.ModuleList(self.hiddenLayers)  # <--- causes DRAMATIC slowdown!

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.relu(out)

        for layer in self.hiddenLayers:
            out = layer(out)
            out = self.relu(out)

        out = self.outputLayer(out)

        return out


# begin training
hidden_width = 128
num_hidden = 4
num_epochs = 5
learning_rate = 0.002

model = AdderNet(num_hidden, hidden_width).to(device)

lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print model.forward(torch.Tensor([2,2]))

train_start_time = time.time()
for epoch in range(num_epochs):
    avgLoss = torch.tensor([0.0], device=device).item()
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

    print(model(torch.tensor([float(500), float(1000)], device=device)))
    print("avg loss: ", avgLoss / len(data))
print("--- %s seconds ---" % (time.time() - train_start_time))


# test
avgLoss = 0
test_size = 50
for i in range(0,test_size):
    a = float(random.randint(A_MIN, A_MAX))
    b = float(random.randint(B_MIN, B_MAX))
    if b == 0:
        b = 1
    x = torch.tensor([a,b], device=device)
    y = torch.tensor([a/b], device=device)
    out = model(x)
    loss = abs(out.item() - y.item())
    avgLoss += loss

    print("test:", a,b, a/b, model(x))


avgLoss /= test_size
print("test loss:", avgLoss)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())