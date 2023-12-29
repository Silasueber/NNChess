import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stockfishHelper import initalizeStockfish


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
def transformSingleBoardToOneHot(board):
    newBoardRepresentation = np.array([board[0]]) # First entry is whos turn it is
    for field in board[1:]:
        newBoardRepresentation = np.append(newBoardRepresentation, one_hot_mapping[field])

    return newBoardRepresentation
def transformBoardsCsvToOneHot(boards):

    newBoardsRepresentation = np.array([])
    for board in boards:
        newBoardRepresentation = transformSingleBoardToOneHot(board)
        newBoardsRepresentation = np.append(newBoardsRepresentation, newBoardRepresentation)

    newBoardsRepresentation = newBoardsRepresentation.reshape(len(boards), 641) #641 = 1+64*10 because one hot vector has 10 elements
    return newBoardsRepresentation

train = True
modelName = "models/p2.pt"

if train:
    dataset = np.loadtxt('data/p4_one_hot.csv', delimiter=',') #TODO represent csv as one hot already?
    X = dataset[:, :65]
    y = dataset[:, 65:]
    X = transformBoardsCsvToOneHot(X)
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # define model
    noOfCpawnValues = 5
    model = nn.Sequential(
        nn.Linear(1 + 64 * noOfCpawnValues * 2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Sigmoid())
    # load model: model = torch.load("models/p2.pt")

    loss_fn = nn.CrossEntropyLoss()  # cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

    batch_size = 100
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_epochs = 1
    for epoch in range(n_epochs):
        for Xbatch, ybatch in dataloader:
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss after each epoch
        print(f'Finished epoch {epoch}, latest loss {loss}')
    torch.save(model, modelName)
    model = torch.load(modelName)
    model.eval()
else:
    # LOAD OLD MODEL AND TRAIN ON IT
    model = torch.load(modelName)

whiteWinning = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
blackWinning = "5kq1/8/8/8/8/8/5K2/8 w - - 0 1"

def convertPositionToString(fen):
    stock = initalizeStockfish()
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
    test = transformSingleBoardToOneHot(test)
    test = torch.tensor(test, dtype=torch.float32)
    predictions = model(test)
    print(predictions)

testFenPosition(whiteWinning)
testFenPosition(blackWinning)