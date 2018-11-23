import torch
import torch.nn as nn

import random
import time

A_MIN = -100
A_MAX = 100
B_MIN = -100
B_MAX = 100

BATCH_SIZE = 1024
TRAIN_SET_BATCHES = 8
TRAIN_SET_SIZE = BATCH_SIZE * TRAIN_SET_BATCHES

NUM_TEST = 10

HIDDEN_WIDTH = 32
HIDDEN_LAYERS = 4
LEARNING_RATE = 0.1
LR_DECAY = 6  # of times LR decays

NUM_EPOCHS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Using device:", device, "LR:"+str(LEARNING_RATE))

# create training data
dx = []
dy = []
for i in range(TRAIN_SET_BATCHES):
    dx_row = []
    dy_row = []
    for j in range(BATCH_SIZE):
        a = float(random.randint(A_MIN,A_MAX) / float(A_MAX))
        b = float(random.randint(B_MIN,B_MAX) / float(B_MAX))
        if b == 0.0:
            b = 1.0 / B_MAX
        dx_row.append([a,b])
        dy_row.append([a/b])

    dx.append(torch.tensor(dx_row))
    dy.append(torch.tensor(dy_row))


class DividerNet(nn.Module):
    def __init__(self, num_hidden, hidden_width):
        super(DividerNet, self).__init__()
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
        if epoch != 0 and epoch % int(NUM_EPOCHS / LR_DECAY) == 0:
            lr *= 0.1
            print("learning rate:",lr)
        param_group['lr'] = lr



# begin training
hidden_width = HIDDEN_WIDTH
num_hidden = HIDDEN_LAYERS
learning_rate = LEARNING_RATE

model = DividerNet(num_hidden, hidden_width)
if torch.cuda.is_available():
    model = model.cuda()

lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

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

    if epoch != 0 and epoch % int(NUM_EPOCHS / 10) == 0 or epoch == 1:
        print("Loss @ Epoch "+ str(epoch) +":", epoch_loss / TRAIN_SET_BATCHES)
print("--- %s seconds ---" % (time.time() - train_start_time))


# test
# for test in range(NUM_TEST):
#     a = float(random.randint(A_MIN, A_MAX) / float(A_MAX))
#     b = float(random.randint(B_MIN, B_MAX) / float(B_MAX))
#
#     print("TEST #"+str(test)+":"+str(A_MAX*a)+"+"+str(B_MAX*b)+"="+str(A_MAX*model(torch.tensor([a,b])).item()))