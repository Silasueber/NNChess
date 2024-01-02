import chess
import random
import platform
from stockfish import Stockfish
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

positions = []
model = "0147_loss.pt"


def progress_bar(tag, current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'{tag}: [{arrow}{padding}] {int(fraction*100)}%', end=ending)


def convertPositionToString(board):
    whites_turn = board.turn
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 1000}

    lines = str(board).replace(" ", "").split('\n')

    result = []
    for line in lines:
        for char in line:
            char = char.strip()
            if char.lower() in piece_values:
                value = piece_values[char.lower()]
                result.append(str(value) if char.islower() else str(-value))
            else:
                result.append('0')
    if whites_turn:
        return "0,"+','.join(result)
    return "1,"+','.join(result)


def createRandomFen(min_moves=10, max_moves=30):
    board = chess.Board()
    num_moves = random.randint(min_moves, max_moves)
    for i in range(num_moves):
        progress_bar("Play Move", i, num_moves)
        if board.is_game_over():
            break
        try:
            legal_moves = [move for move in board.legal_moves]
            random_move = random.choice(legal_moves)
            board.push(random_move)
            pos = (convertPositionToString(
                board)+","+str(getCpawnValue(board.fen()))).split(",")
            pos = [float(p) for p in pos]
            positions.append(pos)
        except Exception as e:
            print(e)
    fen_position = board.fen()
    return fen_position


# i get an error if i try to import it, therefore i copied it here
def initializeStockfish():
    if platform.system() == 'Windows':
        return Stockfish(
            path="C:\\Uni\\Siena_Studium\\Neural Nets\\projects\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")
    else:
        return Stockfish()


stockfish_black = initializeStockfish()


def getCpawnValue(fen):
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    stockfish_black.set_fen_position(fen)
    eval = stockfish_black.get_evaluation()
    eval_type = eval.get('type')
    eval_value = round((eval.get("value") + 700)/1400*0.8+0.1, 4)

    if eval_value > 0.9:
        eval_value = 0.9
    if eval_value < 0.1:
        eval_value = 0.1
    if eval_type == "mate":
        if eval.get('value') > 0:
            return round(0.9+(0.1/eval.get("value")), 4)
        else:
            # Winning Black
            return round(0.1-(0.1/(-eval.get("value"))), 4)
    else:
        return eval_value


def testFenPosition(fen):
    board = chess.Board(fen)
    test = convertPositionToString(board).split(",")
    test = [int(t) for t in test]
    test = torch.tensor(test, dtype=torch.float32)
    predictions = model(test)
    print(predictions)


model_name = "0147_loss"

whiteWinning = "3k4/8/3K4/8/8/8/8/Q7 w - - 0 2"
blackWinning = "8/2k5/8/2q5/8/8/7R/7K w - - 0 1"


for x in range(100):
    positions = []
    for i in range(10):
        progress_bar("Create FEN", i, 10)
        createRandomFen(min_moves=300, max_moves=300)
    positions = np.array(positions)
    X = positions[:, :65]
    y = positions[:, 65:]
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # load model:
    model = torch.load("models/"+model_name+".pt")
    print("load model " + model_name)

    loss_fn = nn.MSELoss()  # BECAUSE ALL LABELS ARE EITHER 0 OR 1
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer

    batch_size = 10
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 2
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
    model_name += str(x)
    torch.save(model, "models/"+model_name+".pt")
    model = torch.load("models/"+model_name+".pt")
    model.eval()

    testFenPosition(whiteWinning)
    testFenPosition(blackWinning)
