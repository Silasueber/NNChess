import chess.pgn
import torch
import random
import argparse
from reinforcement import transformSingleBoardToOneHot, get_highest_legal_q_value_from_predictions
from stockfishHelper import initializeStockfish
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?",
                    help="Which model to play against (default: reinforcment.pt)")
parser.add_argument("--turns", nargs="?",
                    help="How many turns should they each play? (default 20 each)")
args, unknown = parser.parse_known_args()

# Initialize arguments and parameters
if args.model is not None:
    model = torch.load(f"models/{args.model}.pt")
else:
    model = torch.load("models/reinforcement.pt")

if args.turns is not None:
    turns_to_be_played = int(args.turns)
else:
    turns_to_be_played = 20


# Setup board
initial_fen = "2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1"
board = chess.Board(initial_fen)
game = chess.pgn.Game()
game.headers["FEN"] = initial_fen


def printBoard():
    """
    prints the board to console
    :return:
    """
    print("a b c d e f g h")
    print("---------------")
    print(board)

def getBestMove():
    """
    Lets the model predict Q-Values and makes the best legal play possible
    :return: Best legal play possible as by prediction of the model
    """
    board_one_hot = transformSingleBoardToOneHot(board)
    X = torch.tensor(board_one_hot, dtype=torch.float32)
    #predict q-values
    action_q_values = model(X)
    # select highest legal q value
    move = get_highest_legal_q_value_from_predictions(board, action_q_values)
    if move is None:
    # occurs only if legal_moves has a single move? just play that move then?
        return list(board.legal_moves)[0]
    return move

# 5. Game Loop

wins = 0
draws = 0
no_of_games = 10000
for game_number in range(no_of_games):
    turns_played = 0
    board = chess.Board(initial_fen)
    game = chess.pgn.Game()
    game.headers["FEN"] = initial_fen
    node = game
    while not board.is_game_over() and turns_played < turns_to_be_played:
        turns_played += 1
        # Bot move
        move = getBestMove()
        node = node.add_variation(move)
        print(f"board fen: {board.fen()}")
        print(f"move: {move}")
        # in rare occasions the move is None because the
        board.push(move)
        # Random move
        if not board.is_game_over():
            legal_moves = [move for move in board.legal_moves]
            random_move = random.choice(legal_moves)
            node = node.add_variation(random_move)
            board.push(random_move)
    fen_position = board.fen()
    evaluator = initializeStockfish()
    evaluator.set_fen_position(fen_position)
    evaluation = evaluator.get_evaluation().get("value")
    print(evaluation)
    if evaluation > 0:
        wins += 1
    elif evaluation == 0:
        draws += 1

print(wins)
print(f"wins: {wins}, draws: {draws},  win percentage: {wins/(no_of_games-draws)}")
# Print and save the PGN
# print(game)
# print(" ")
# print("open https://www.chess.com/analysis?tab=analysis -> paste output in 'Load From FEN/PGN(s)'")
# fen_position = board.fen()
# evaluator = initializeStockfish()
# evaluator.set_fen_position(fen_position)
# print("Interpretation of evaluation: \n\tPositive values --> White ahead\n\tNegative values --> Black ahead")
# print(f"FEN-Position: {fen_position}")
# print(f"Stockfish evaluation of situation: {evaluator.get_evaluation().get('value')}")