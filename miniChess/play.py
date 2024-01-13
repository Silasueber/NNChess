import chess
import chess.pgn
import torch
# from dataCreation import convertPositionToString

# 1. Init model
model = torch.load("models/minichess.pt")

# 2. Init chess Board
initial_fen = "2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1"
board = chess.Board(initial_fen)
game = chess.pgn.Game()
game.headers["FEN"] = initial_fen
# 3. function to eval Position


def convertPositionToString(board):
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 100}
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


def evalPosition():
    # little cheating
    moves = list(board.legal_moves)
    if len(moves) == 0:
        if board.is_checkmate():
            if board.turn:
                return -100
            else:
                return 100
        return 0.5

    whites_turn = board.turn
    position = ("1," + convertPositionToString(board)
                if whites_turn else "0," + convertPositionToString(board))
    position = position.split(",")
    position = [int(pos) for pos in position]
    postition_eval = torch.tensor(position, dtype=torch.float32)
    predictions = model(postition_eval)
    # -0.5 because we need to establish postive and negativ values for the chess bot
    return predictions[0].item() - 0.5

# 4. function to look which move is the best


# def minimax(depth):
#     if depth == 0:
#         return evalPosition()

#     moves = list(board.legal_moves)
#     # Either Check or Stalemate
#     if len(moves) == 0:
#         if board.is_checkmate():
#             return -1 - (0.1*depth)
#         return 0.5
#     value = -2
#     for move in moves:
#         board.push(move)
#         eval = -minimax(depth-1)
#         board.pop()
#         value = max(eval, value)
#     return value


def getBestMove():
    bestMove = None
    # Values will be between -2 and +2
    bestValue = -10 if board.turn else 10
    moves = list(board.legal_moves)
    for move in moves:
        board.push(move)
        # value = -minimax(2)
        value = evalPosition()
        if (not board.turn and value > bestValue) or (board.turn and value < bestValue):
            bestValue = value
            bestMove = move
        board.pop()
    print(bestValue)
    return bestMove


# 5. Game Loop
counter = 1
node = game

while not board.is_game_over():
    move = getBestMove()
    node = node.add_variation(move)
    board.push(move)
    counter += 1

# Print and save the PGN
print(game)
