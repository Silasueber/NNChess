import csv
import chess
from tools import convertPositionToString, initializeStockfish
import random
import argparse
import math

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--amount", default=10, type=int, nargs="?",
                    help="Amount of games (Default: 10)")
parser.add_argument("--random", default=0.5, type=float, nargs="?",
                    help="Random move instead of best move (Default: 0.5)")
parser.add_argument("--name", default="minichess.csv", nargs="?",
                    help="Name of the save (Default: minichess.csv)")
parser.add_argument("--position", default="2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1", nargs="?",
                    help="Start position of the chess game (Default: 2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1)")
args = parser.parse_args()

# Set hyperparameters
amount_of_games = int(args.amount)
random_moves = float(args.random)
name = args.name
position = args.position

# Init Stockfish parameters
position_evaluated = []
stockfish_white = initializeStockfish()
stockfish_black = initializeStockfish()

board = chess.Board()

draw_counter = 0

# Play moves


def playMove(move):
    stockfish_white.make_moves_from_current_position([move])
    stockfish_black.make_moves_from_current_position([move])
    board.push(chess.Move.from_uci(str(move)))


def createDataEntry():
    if board.fen() not in position_evaluated:
        position_evaluated.append(board.fen())
        position = convertPositionToString(board)
        print(board)
        # Dataset three with cpawn value for position [limited at -10 and +10]
        csv_path = name
        winner = getCpawnValue()
        # if (winner < 0.5 and winner > 0.1) or (winner > 0.5 and winner < 0.9):
        line = position+","+str(winner)
        try:
            with open(csv_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(line.split(','))
        except Exception as e:
            print(f"Error: {e}")


def getCpawnValue():
    global draw_counter
    eval = stockfish_black.get_evaluation()
    eval_type = eval.get('type')
    if eval_type == "mate":
        if eval.get("value") > 0:
            return 1
        else:
            return 0
    K = 10
    try:
        eval_value = 1/(1+(math.pow(10, -(eval.get("value")/K))))
    except OverflowError:
        eval_value = 1
    if eval_value == 0.5:
        draw_counter += 1
    return eval_value


def createRandomFen(num_moves):
    board = chess.Board(position)
    for i in range(num_moves):
        if board.is_game_over():
            break
        try:
            legal_moves = [move for move in board.legal_moves]
            random_move = random.choice(legal_moves)
            board.push(random_move)
        except:
            print("Couldnt make move")
    fen_position = board.fen()
    print(fen_position)
    return fen_position


def playGame(random_moves=0.5, moves=5):
    global draw_counter
    draw_counter = 0
    while moves > 0 and stockfish_white.get_best_move():
        moves -= 1
        # Stop Games with dead draws
        if draw_counter > 10:
            break
        try:
            if random.random() > random_moves:
                # board.turn True if it is whites turn
                if board.turn:
                    createDataEntry()
                    playMove(stockfish_white.get_best_move())
                else:
                    createDataEntry()
                    playMove(stockfish_black.get_best_move())
            else:
                # board.turn True if it is whites turn
                if board.turn:
                    createDataEntry()
                    legal_moves = [move for move in board.legal_moves]
                    random_move = random.choice(legal_moves)
                    playMove(random_move)
                else:
                    createDataEntry()
                    legal_moves = [move for move in board.legal_moves]
                    random_move = random.choice(legal_moves)
                    playMove(random_move)
        except Exception as e:
            print(e)


def setPosition(position):
    stockfish_black.set_fen_position(position)
    stockfish_white.set_fen_position(position)
    board.set_fen(position)


for i in range(100000):
    print("Current Game: " + str(i+1))
    for x in range(5):
        setPosition(createRandomFen(x))
        playGame(random_moves=random_moves)
