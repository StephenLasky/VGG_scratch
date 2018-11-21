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

BATCH_SIZE = 1024 * 2
TRAIN_SET_BATCHES = 4
TRAIN_SET_SIZE = BATCH_SIZE * TRAIN_SET_BATCHES

HIDDEN_WIDTH = 256
HIDDEN_LAYERS = 16
LEARNING_RATE = 0.01

NUM_EPOCHS = 5000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Using device:", device)

# create training data
dx = []
dy = []
for i in range(TRAIN_SET_BATCHES):
    dx_row = []
    dy_row = []
    for j in range(BATCH_SIZE):
        a = float(random.randint(A_MIN,A_MAX))
        b = float(random.randint(B_MIN,B_MAX))
        if b == 0:
            b = 1
        dx_row.append([a,b])
        dy_row.append([a+b])

    dx.append(torch.tensor(dx_row))
    # dy_row = torch.tensor(dy_row)
    dy.append(torch.tensor(dy_row))






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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        # MIN = 0.001
        # if lr < MIN:
        #     lr = MIN
        if epoch % 500 == 0:
            lr *= 0.5
            print("learning rate:",lr)
        param_group['lr'] = lr



# begin training
hidden_width = HIDDEN_WIDTH
num_hidden = HIDDEN_LAYERS
learning_rate = LEARNING_RATE

model = AdderNet(num_hidden, hidden_width)
if torch.cuda.is_available():
    model = model.cuda()

lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

# print model.forward(torch.Tensor([2,2]))


# train_start_time = time.time()
# for epoch in range(num_epochs):
#     avgLoss = torch.tensor([0.0], device=device).item()
#     for i in range(0,len(data)):
#         d = data[i]
#         x = d.x
#         y = d.y
#
#         out = model.forward(x)
#         loss = lossFunction(out, y)
#         avgLoss += loss.item()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(model(torch.tensor([float(500), float(1000)], device=device)))
#     print("avg loss: ", avgLoss / len(data))
# print("--- %s seconds ---" % (time.time() - train_start_time))

train_start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for batch in range(TRAIN_SET_BATCHES):
        x = dx[batch]
        y = dy[batch]

        out = model.forward(x)
        loss = lossFunction(out,y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    adjust_learning_rate(optimizer, epoch)

    if epoch % 100 == 0:
        print("AVG Epoch loss:", epoch_loss / TRAIN_SET_BATCHES)
print("--- %s seconds ---" % (time.time() - train_start_time))


# test
