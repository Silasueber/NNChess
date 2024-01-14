import chess
import chess.pgn
import torch
import random
import argparse
import sys
from reinforcement import convertPositionToString, transformSingleBoardToOneHot, get_highest_legal_q_value_from_predictions

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--play", nargs="?",
                    help="To determine if you want to play against the NN (y or n) (Default: n)")
parser.add_argument("--model", nargs="?",
                    help="Which model to play against")
args = parser.parse_args()

if args.model != None:
    # 1. Init model
    model = torch.load(f"models/{args.model}.pt")
else:
    model = torch.load("models/reinforcement.pt")
    # print("Please provide a model (--model)")
    # sys.exit()

# 2. Init chess Board
initial_fen = "2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1"
board = chess.Board(initial_fen)
game = chess.pgn.Game()
game.headers["FEN"] = initial_fen
# 3. function to eval Position


def printBoard():
    print("a b c d e f g h")
    print("---------------")
    print(board)

def getBestMove():
    board_one_hot = transformSingleBoardToOneHot(board)
    X = torch.tensor(board_one_hot, dtype=torch.float32)
    #predict move
    action_q_values = model(X)
    # select highest q value
    move = get_highest_legal_q_value_from_predictions(board, action_q_values)
    return move

# 5. Game Loop
node = game

if args.play != None:
    if args.play == "y":
        play = True
    else:
        play = False
else:
    play = False

while not board.is_game_over():
    # Bot move
    move = getBestMove()
    node = node.add_variation(move)
    board.push(move)

    if play:
        # Human move
        printBoard()
        if not board.is_game_over():
            correct_move = False
            while not correct_move:
                move = input("Your move (ex. e2e4): ")
                try:
                    board.push_uci(move)
                    node = node.add_variation(chess.Move.from_uci(str(move)))
                    correct_move = True
                    printBoard()
                except:
                    print("Invalid move!")

    else:
        # Random move
        if not board.is_game_over():
            legal_moves = [move for move in board.legal_moves]
            random_move = random.choice(legal_moves)
            node = node.add_variation(random_move)
            board.push(random_move)


# Print and save the PGN
print(game)
print(" ")
print("open https://www.chess.com/analysis?tab=analysis -> paste output in 'Load From FEN/PGN(s)'")