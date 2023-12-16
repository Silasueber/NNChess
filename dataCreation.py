import chess
import os

from stockfish import Stockfish

# Init Stockfish parameters
stockfish_white = Stockfish()
stockfish_white.set_elo_rating(180)
stockfish_black = Stockfish()
# stockfish_black.set_elo_rating(1350)

# Play moves
def playMove(move):
    stockfish_white.make_moves_from_current_position([move])
    stockfish_black.make_moves_from_current_position([move])

def convertPositionToString(board):
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


# two neurons for determine whose turn it is white->(1,0) black->(0,1) 
def createDataEntry(whitesTurn):
    position = convertPositionToString(stockfish_black.get_board_visual())
    eval = stockfish_black.get_evaluation()
    turn = "1,0" if whitesTurn else "0,1"
    print(turn+","+position+",")

while stockfish_white.get_best_move():
    playMove(stockfish_white.get_best_move())
    os.system('clear')
    print(stockfish_white.get_board_visual())
    playMove(stockfish_black.get_best_move())
    os.system('clear')
    print(stockfish_white.get_board_visual())