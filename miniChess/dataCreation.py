import csv
import chess
from stockfishHelper import initializeStockfish
import random
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--amount", nargs="?",
                    help="Amount of games (Default: 10)")
parser.add_argument("--random", nargs="?",
                    help="Random move instead of best move (Default: 0.5)")
parser.add_argument("--position", nargs="?",
                    help="Start position of the chess game (Default: 2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1)")
args = parser.parse_args()


# Init Stockfish parameters
stockfish_white = initializeStockfish()
stockfish_black = initializeStockfish()

stockfish_eval = initializeStockfish()

board = chess.Board()

draw_counter = 0

# Play moves


def playMove(move):
    stockfish_white.make_moves_from_current_position([move])
    stockfish_black.make_moves_from_current_position([move])
    try:
        board.push(move)
    except:
        board.push_uci(move)
    print(stockfish_black.get_board_visual())


def convertPositionToString(board):
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 100}

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


def createDataEntry(whitesTurn):
    position = convertPositionToString(stockfish_black.get_board_visual())
    turn = "1," if whitesTurn else "0,"
    # Dataset three with cpawn value for position [limited at -10 and +10]
    csv_path = "eval/miniChess.csv"
    winner = getCpawnValue()
    line = turn+position+","+str(winner)
    try:
        with open(csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(line.split(','))
    except Exception as e:
        print(f"Error: {e}")


def getCpawnValue():
    global draw_counter
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    eval = stockfish_black.get_evaluation()
    eval_type = eval.get('type')
    eval_value = round((eval.get("value") + 700)/1400*0.8+0.1, 4)
    if eval_value == 0.5:
        draw_counter += 1
    if eval_value > 0.9:
        eval_value = 0.9
    if eval_value < 0.1:
        eval_value = 0.1
    if eval_type == "mate":
        if eval.get('value') > 0:
            return round(0.9+(0.1/eval.get("value")), 4)
        else:
            # Winning Black
            if eval.get('value') == 0:
                return round(0.1-(0.1/(-0.00001)), 4)
            return round(0.1-(0.1/(-eval.get("value"))), 4)
    else:
        return eval_value


def playGame(random_moves=0.5):
    global draw_counter
    draw_counter = 0
    while stockfish_white.get_best_move():
        # Stop Games with dead draws
        if draw_counter > 5:
            break
        try:
            if random.random() < random_moves:
                if board.turn:
                    createDataEntry(whitesTurn=True)
                    playMove(stockfish_white.get_best_move_time(100))
                else:
                    createDataEntry(whitesTurn=False)
                    playMove(stockfish_black.get_best_move_time(100))
            else:
                if board.turn:
                    createDataEntry(whitesTurn=True)
                    legal_moves = [move for move in board.legal_moves]
                    random_move = random.choice(legal_moves)
                    playMove(random_move)
                else:
                    createDataEntry(whitesTurn=False)
                    legal_moves = [move for move in board.legal_moves]
                    random_move = random.choice(legal_moves)
                    playMove(random_move)
        except Exception as e:
            print(e)


def playGameLimitedMoves(moves):
    while moves > 0 and stockfish_white.get_best_move() and stockfish_black.get_best_move():
        try:
            createDataEntry(whitesTurn=True)
            playMove(stockfish_white.get_best_move_time(100))
            createDataEntry(whitesTurn=False)
            playMove(stockfish_black.get_best_move_time(100))

            moves -= 1
        except Exception as e:
            print(e)
            moves = 0


def setPosition(position):
    stockfish_black.set_fen_position(position)
    stockfish_white.set_fen_position(position)
    board.set_fen(position)


if args.amount != None:
    amount_of_games = int(args.amount)
else:
    amount_of_games = 10

if args.random != None:
    random_moves = float(args.random)
else:
    random_moves = 0.5

if args.position != None:
    position = args.position
else:
    position = "2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1"
for i in range(amount_of_games):
    print("Current Game: " + str(i+1))
    setPosition(position)
    playGame(random_moves=random_moves)
