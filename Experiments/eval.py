import torch
from stockfish import Stockfish
import chess
from stockfishHelper import initializeStockfish

# Load Model 
model = torch.load("models/p2.pt")

# Init Stockfish
stockfish = initializeStockfish()

# Init Chess
board = chess.Board()


def convertPositionToString(fen):
    stock = initializeStockfish()
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

def checkMoves():
    current_best = 0
    current_best_move = None
    for move in board.legal_moves:
        board.push(move)
        test = ("1,"+ convertPositionToString(board.fen())).split(",")
        print(convertPositionToString(board.fen()))
        test = [int(t) for t in test]
        test = torch.tensor(test, dtype=torch.float32)
        predictions = model(test)
        print(predictions)
        print(predictions[1].item())
        if predictions[1].item() > current_best:
            current_best = predictions[1].item()
            current_best_move = move
        print(move)
        board.pop()
    return current_best_move


while not board.is_game_over():
    # White Turns
    best_move = stockfish.get_best_move()
    move = chess.Move.from_uci(best_move)
    board.push(move)
    stockfish.make_moves_from_current_position([best_move])
    print(board)
    print("---------------------")
    
    # Black Turns
    move = checkMoves()
    print(move)
    board.push(move)
    stockfish.make_moves_from_current_position([move])
    print(board)
    print("---------------------")

    