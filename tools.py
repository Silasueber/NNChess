from stockfish import Stockfish
from dotenv import load_dotenv
import os
import sys


def initializeStockfish():
    """
    Check if a stockfish path is given in the env file and use it

    :return: Stockfish instance
    """
    # load the enviroment variables
    load_dotenv()
    STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
    if STOCKFISH_PATH == "":
        try:
            return Stockfish()
        except:
            print("Please download stockfish https://stockfishchess.org/download/ and set up the stockfish path in the .env")
            sys.exit()

    else:
        return Stockfish(STOCKFISH_PATH)


def convertPositionToString(board):
    """
    Convert chess board position to a string for model input.

    :param board: The chess board.
    :return: The board as a string encoded.
    """
    whites_turn = board.turn
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 100}
    board_str = str(board)
    lines = board_str.split('\n')

    result = ["1" if whites_turn else "0"]
    for line in lines:
        for char in line.split(' '):
            char = char.strip()
            if char.lower() in piece_values:
                value = piece_values[char.lower()]
                result.append(str(value) if char.islower() else str(-value))
            else:
                result.append('0')
    return ','.join(result)
