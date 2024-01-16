import chess
import torch
from Experiments.chessBot import getBestMove

board = chess.Board("6k1/3Q4/6K1/8/8/8/8/8 b - - 0 1")
model = torch.load("models/p3_2.pt")


start = str(input("Do you want to play with white or black (w/b): "))
while not (start == "w" or start == "b"):
    print("Invalid starting color!")
    print(start)
    start = input("Do you want to play with white or black (w/b): ")

if start == "b":
    board.push(getBestMove(board))

while not board.is_game_over():
    print(board)
    move = input("Your move (ex. e2e4): ")
    try:
        board.push_uci(move)
    except:
        print("Invalid move!")
    board.push(getBestMove(board))
