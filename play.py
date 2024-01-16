import chess
import chess.pgn
import torch
import random
import argparse
import sys
import stockfish
from tools import convertPositionToString, initializeStockfish

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--play", nargs="?",
                    help="Determine if you want to play against the NN (y or n) (Default: y)")
parser.add_argument("--model", required=True, nargs="?",
                    help="Specify the model to play against")
parser.add_argument("--elo", type=int, default=200, nargs="?",
                    help="Set the elo of Stockfish if you don't play against the bot (Default: 200)")
parser.add_argument("--random", nargs="?",
                    help="Play against random moves (y or n) (Default: n)")
parser.add_argument("--depth", type=int, default=2, nargs="?",
                    help="Depth for the minimax algorithm (Default: 2)")
parser.add_argument("--checkmate", default="n", nargs="?",
                    help="Check for checkmate and then ignore model eval (y or n ) (Default: n)")

args = parser.parse_args()

# Set hyperparameters
elo_rating = args.elo
depth = args.depth
play = args.play == "y" if args.play is not None else True
random_moves = args.random == "y" if args.random is not None else False
checkmate = args.checkmate == "y" if args.checkmate is not None else True

# Load the specified model, exit if not provided
try:
    model = torch.load(args.model)
except FileNotFoundError:
    print("Model file not found. Please provide a valid model file using --model.")
    sys.exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit()


# Initialize Stockfish
stockfish = initializeStockfish()
# Set Stockfish elo rating
stockfish.set_elo_rating(elo_rating)

# Initialize chess board
initial_fen = "2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1"
board = chess.Board(initial_fen)
stockfish.set_fen_position(initial_fen)
game = chess.pgn.Game()
game.headers["FEN"] = initial_fen
game.headers["White"] = "Neural Network Bot"
game.headers["Black"] = f"Stockfish {elo_rating}"


def evalPosition():
    """
    Convert board into tensor and evaluate current board position using the neural network model.

    :return: Value between 0.5(White winning) and -0.5(Black winning).
    """
    if len(list(board.legal_moves)) == 0 and checkmate:
        return -50
    position = convertPositionToString(board)
    position = position.split(",")
    position = [int(pos) for pos in position]
    position_eval = torch.tensor(position, dtype=torch.float32)
    predictions = model(position_eval)
    return predictions[0].item() - 0.5


def maximum(depth):
    """
    Get the maximum Value possibible for all the moves

    :param1 depth: the current depth of the minimax algorithm
    :return: the best move and the maximum value
    """
    if depth == 0 or len(list(board.legal_moves)) == 0:
        return None, evalPosition()
    maxValue = -10000
    bestMove = None
    moves = list(board.legal_moves)
    for move in moves:
        board.push(move)
        eval = minimum(depth-1)
        board.pop()
        if eval > maxValue:
            maxValue = eval
            bestMove = move
    return bestMove, maxValue


def minimum(depth):
    """
    Get the minimum value for all possible moves

    :param1 depth: the current depth of the minimax algorithm
    :return: minimum value
    """
    if depth == 0 or len(list(board.legal_moves)) == 0:
        return evalPosition()
    minValue = 10000
    moves = list(board.legal_moves)
    for move in moves:
        board.push(move)
        _, eval = maximum(depth-1)
        board.pop()
        if eval < minValue:
            minValue = eval
    return minValue


def getBestMove():
    """
    Get the best move.

    :return: The best move.
    """
    bestMove, _ = maximum(depth)
    return bestMove


def printBoard():
    """
    Print the chess board on the console with the file names.
    """
    print("a b c d e f g h")
    print("---------------")
    print(board)


node = game
# TODO Delete only for visualize
random_moves = False
nn_wins = 0
random_wins = 0
game_to_play = 100

for i in range(game_to_play):
    print("Playing Game " + str(i))
    board = chess.Board(initial_fen)
    stockfish.set_fen_position(initial_fen)
    node = game
    # Main game loop
    while not board.is_game_over():

        # Bot move
        move = getBestMove()
        printBoard()
        node = node.add_variation(move)
        board.push(move)
        if not random_moves:
            stockfish.make_moves_from_current_position([move])

        if play:
            # Human move
            printBoard()
            if not board.is_game_over():
                correct_move = False
                while not correct_move:
                    move_str = input("Your move (ex. e2e4): ")
                    try:
                        board.push_uci(move_str)
                        move = chess.Move.from_uci(str(move_str))
                        if not random_moves:
                            stockfish.make_moves_from_current_position([move])
                        node = node.add_variation(move)
                        correct_move = True
                        printBoard()
                    except:
                        print("Invalid move!")
        else:
            # Random move
            if not board.is_game_over():
                if random_moves:
                    legal_moves = [move for move in board.legal_moves]
                    random_move = random.choice(legal_moves)
                    node = node.add_variation(random_move)
                    board.push(random_move)
                else:
                    move = stockfish.get_best_move()
                    move = chess.Move.from_uci(str(move))
                    board.push(move)
                    stockfish.make_moves_from_current_position([move])
                    node = node.add_variation(move)
    outcome = board.outcome()
    if outcome:
        if outcome.winner == chess.WHITE:
            nn_wins += 1
            print("win")
        elif outcome.winner == chess.BLACK:
            random_wins += 1
            print("loss")
        else:
            print("draw")
    # Print the PGN
            # TODO REMOVE COMMENTS
    # print(game)
    # print(" ")
    # print("Open https://www.chess.com/analysis?tab=analysis -> paste output in 'Load From FEN/PGN(s)'")

print(nn_wins, random_wins)
