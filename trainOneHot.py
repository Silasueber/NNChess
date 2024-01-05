import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import os.path as path
from torch.utils.data import TensorDataset, DataLoader
# load the dataset, split into input (X) and output (y) variables
# BE CAREFUL the split must be changed if we use a different representation of who is winning cpawn vs, [1,0] 
one_hot_mapping = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # White Pawn
    3: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # White Knight/Bishop
    5: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # White Rook
    10: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # White Queen
    1000: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # White King
    -1: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Black Pawn
    -3: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Black Knight/Bishop
    -5: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Black Rook
    -10: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Black Queen
    -1000: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Black King
}
train = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelName = "models/one_hot.pt"

def transformSingleBoardToOneHot(board):
    newBoardRepresentation = np.array([board[0]]) # First entry is whose turn it is
    for field in board[1:]:
        newBoardRepresentation = np.append(newBoardRepresentation, one_hot_mapping[field])

    return newBoardRepresentation
def transformBoardsCsvToOneHot(boards):
    oneHotEncodedValuesFileName = "data/p2_one_hot_encoded.npy"
    if path.isfile(oneHotEncodedValuesFileName):
        with open(oneHotEncodedValuesFileName, 'rb') as f:
            return np.load(f)
    newBoardsRepresentation = np.array([])
    for board in boards:
        newBoardRepresentation = transformSingleBoardToOneHot(board)
        newBoardsRepresentation = np.append(newBoardsRepresentation, newBoardRepresentation)

    newBoardsRepresentation = newBoardsRepresentation.reshape(len(boards), 641) #641 = 1+64*10 because one hot vector has 10 elements
    with open(oneHotEncodedValuesFileName, "wb") as f:
        np.save(f, newBoardsRepresentation)
    return newBoardsRepresentation


if train:
    print('Using device:', device)
    dataset = np.loadtxt('data/p2_small.csv' if device.type == 'cpu' else 'data/p2.csv', delimiter=',') # use same dataset because no reason to change
    X = dataset[:, :65]
    y = dataset[:, 65:]
    X = transformBoardsCsvToOneHot(X)
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y_one_hot = torch.tensor(y, dtype=torch.float32)
    y_class_indices = torch.argmax(y_one_hot, dim=1) #pytorch needs integer values for nn.CrossEntropyLoss

    # define model
    noOfCpawnValues = 5
    model = nn.Sequential(
        nn.Linear(1 + 64 * noOfCpawnValues * 2, 512),
        nn.ReLU(),
        nn.Linear(512, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3))

    # load model:
    if path.isfile(modelName):
        model = torch.load(modelName, map_location=device)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss() # TODO still not sure? habs auch nochmal gegoogelt
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer
    batch_size = 100
    dataset = TensorDataset(X, y_class_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 2000
    for epoch in range(n_epochs):
        avg_loss = 0
        amount = 0
        for Xbatch, ybatch in dataloader:
            Xbatch, ybatch = Xbatch.to(device), ybatch.to(device)
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            avg_loss += loss.item()
            amount += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print loss after each epoch
        print(f'Finished epoch {epoch}, latest loss {str(avg_loss/amount)}')

    torch.save(model, modelName)
    model = torch.load(modelName)
    model.eval()
else:
    model = torch.load(modelName, map_location=device)

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
    test = transformSingleBoardToOneHot(test)
    test = torch.tensor(test, dtype=torch.float32)
    test.to(device)
    predictions = model(test)
    print(predictions)

with torch.no_grad(): # uses less memory, random optimization when doing inference
    testFenPosition(whiteWinning)
    testFenPosition(blackWinning)