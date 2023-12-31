import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import time
from torch.utils.data import TensorDataset, DataLoader
# load the dataset, split into input (X) and output (y) variables
# BE CAREFUL the split must be changed if we use a different representation of who is winning cpawn vs, [1,0] 

train = False
if train:
    dataset = np.loadtxt('data/p3_2.csv', delimiter=',')
    X = dataset[:, :65]
    y = dataset[:, 65:]
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # define model
    model = nn.Sequential(
        nn.Linear(65, 512),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1))

    # load model: 
    model = torch.load("models/018_loss.pt")

    loss_fn = nn.MSELoss()  # BECAUSE ALL LABELS ARE EITHER 0 OR 1
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

    batch_size = 10
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 2000
    for epoch in range(n_epochs):
        avg_loss = 0
        amount = 0
        for Xbatch, ybatch in dataloader:
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            avg_loss += loss.item()
            amount += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print loss after each epoch
        print(f'Finished epoch {epoch}, latest loss {str(avg_loss/amount)}')

    torch.save(model, "models/3_layer_4.pt")
    model = torch.load("models/3_layer_4.pt")
    model.eval()
else:
    model = torch.load("models/3_layer_4.pt")

whiteWinning = "3k4/8/3K4/8/8/8/8/Q7 w - - 0 2"
blackWinning = "8/2k5/8/2q5/8/8/7R/7K w - - 0 1"

def convertPositionToString(fen):
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 1000}
    board = chess.Board(fen)
    board = str(board)
    lines = board.split('\n')
   
    result = []
    for line in lines:
        for char in line.split(' '):
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