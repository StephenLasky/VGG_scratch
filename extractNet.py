# a brute force attempt at extracting a substring from test

import torch
import torch.nn as nn

import random
# random.seed(0)
import time

MAX_INPUT_STR = 32
PAD_CHAR = '.'

BATCH_SIZE = 1024 * 4
TRAIN_SET_BATCHES = 16
TRAIN_SET_SIZE = BATCH_SIZE * TRAIN_SET_BATCHES
NUM_TEST = 10

NUM_EPOCHS = 500

HIDDEN_WIDTH = 4
HIDDEN_LAYERS = 8
LEARNING_RATE = 0.1
LR_DECAY = 6  # of times LR decays

def printChrList(x):
    out = ""
    for y in x:
        out += chr(y)
    return out

def normalize(x):
    min = 96
    max = 122
    avg = min + (max-min) / 2
    rng = float(max - min) / 2

    ret = (float(x) - avg) / rng

    # print("IN:"+str(x)+" OUT:"+str(ret))

    return ret

def denormalize(x):
    min = 96
    max = 122
    avg = min + (max - min) / 2
    rng = float(max - min) / 2

    x *= rng
    x += avg
    x = round(x)
    x = int(x)

    return x

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if epoch != 0 and epoch % int(NUM_EPOCHS / LR_DECAY) == 0:
            lr *= 0.1
            print("learning rate:",lr)
        param_group['lr'] = lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')




# generate training data
dx = []
dy = []
for i in range(TRAIN_SET_BATCHES):
    dx_row = []
    dy_row = []
    for j in range(BATCH_SIZE):

        charsX = []
        charsY = []
        for c in range(MAX_INPUT_STR):
            char = random.randint(ord('a') - 1, ord('z'))
            if char != ord('a') - 1:
                charsX.append(normalize(char))
                charsY.append(normalize(char))
            else:
                charsX.append(normalize(ord('a') - 1))

        while len(charsY) < MAX_INPUT_STR:
            charsY.append(normalize(ord('a') - 1))

        dx_row.append(charsX)
        dy_row.append(charsY)

    dx.append(torch.tensor(dx_row))
    dy.append(torch.tensor(dy_row))


# print(ord('a') - 1)
# print(ord('a'))
# print(ord('z'))

class ExtractNet(nn.Module):
    def __init__(self, input_width, hidden_width, hidden_layers):
        super(ExtractNet, self).__init__()
        self.relu = nn.ReLU()

        self.hiddenLayers = []
        self.inputLayer = nn.Linear(input_width, hidden_width)
        self.outputLayer = nn.Linear(hidden_width, input_width)

        for i in range(hidden_layers):
            self.hiddenLayers.append(nn.Linear(hidden_width, hidden_width))

    def forward(self, x):
        out = self.inputLayer(x)
        out = self.relu(out)

        for layer in self.hiddenLayers:
            out = layer(out)
            out = self.relu(out)

        out = self.outputLayer(out)

        return out


model = ExtractNet(MAX_INPUT_STR, 32, 1)
if torch.cuda.is_available():
    model = model.cuda()
lossFunction = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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







# short test
for t in range(NUM_TEST):
    charsX = []
    charsXInput = []
    charsY = []
    for c in range(MAX_INPUT_STR):
        char = random.randint(ord('a') - 1, ord('z'))
        if char != ord('a') - 1:
            charsXInput.append(char)
            charsX.append(normalize(char))
            charsY.append(char)
        else:
            charsXInput.append(ord('a') - 1)
            charsX.append(ord('a') - 1)

    while len(charsY) < MAX_INPUT_STR:
        charsY.append(ord('a') - 1)

    # print "GUESS:" + printChrList(charsXInput) + '\n  GOT:' + printChrList(charsY) + '\n'