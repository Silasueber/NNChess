import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stockfish import Stockfish
from torch.utils.data import TensorDataset, DataLoader
# load the dataset, split into input (X) and output (y) variables
# BE CAREFUL the split must be changed if we use a different representation of who is winning cpawn vs, [1,0] 
dataset = np.loadtxt('data/p2.csv', delimiter=',')
X = dataset[:, :65]
y = dataset[:, 65:]
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# define model
model = nn.Sequential(
    nn.Linear(65, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    nn.Sigmoid())

loss_fn = nn.CrossEntropyLoss()  # cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

batch_size = 100
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# LOAD OLD MODEL AND TRAIN ON IT
model = torch.load("models/p2.pt")
n_epochs = 2000
if True:
    for epoch in range(n_epochs):
        for Xbatch, ybatch in dataloader:
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss after each epoch
        print(f'Finished epoch {epoch}, latest loss {loss}')

    torch.save(model, "models/p2.pt")
model = torch.load("models/p2.pt")
model.eval()


whiteWinning = "4k3/8/2n5/8/8/5QK1/8/8 w - - 0 1"
blackWinning = "3qkr2/8/2n5/8/8/5QK1/8/8 b - - 0 1"

def convertPositionToString(fen):
    stock = Stockfish()
    stock.set_fen_position(fen)
    board = stock.get_board_visual()

    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 1000}

    lines = board.split('\n')[1:-1]
    result = []
    for line in lines:
        for char in line[2:-2].split('|')[:-1]:
            char = char.strip()
            if char.lower() in piece_values:
                value = piece_values[char.lower()]
                result.append(str(value) if char.islower() else str(-value))
            else:
                result.append('0')

    return ','.join(result)

def testFenPosition(fen):
    test = ("1,"+ convertPositionToString(fen)).split(",")
    test = [int(t) for t in test]
    test = torch.tensor(test, dtype=torch.float32)
    predictions = model(test)
    print(predictions)

testFenPosition(whiteWinning)
testFenPosition(blackWinning)