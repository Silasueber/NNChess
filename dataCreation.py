import chess
import csv
import random
from stockfish import Stockfish

# Init Stockfish parameters
stockfish_white = Stockfish()
stockfish_white.set_elo_rating(1000)
stockfish_black = Stockfish()
# stockfish_black.set_elo_rating(1350)

csv_path = "positions.csv"

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
    eval = convertEvalIntoValue()
    turn = "1,0," if whitesTurn else "0,1,"
    line = turn+position+","+str(eval)
    try:
        with open(csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(line.split(','))
        print(f"Line added to {csv_path} successfully.")
    except Exception as e:
        print(f"Error: {e}")



def convertEvalIntoValue():
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    eval = stockfish_black.get_evaluation()
    eval_type = eval.get('type')
    eval_value = eval.get("value")
    if eval_type == "mate":
        if eval_value > 0: 
            # Winning white
            return 1
        else: 
            # Winning Black
            return -1
    else:
        if eval_value > 100: 
            # Winning white
            return 1
        elif eval_value < -100: 
            # Winning Black
            return -1
        else:
            # Equal
            return 0
    
    # Can be used later for exact values
        
    # if eval.get('type') == "cp":
    #     print(eval.get('value'))
    # else:
    #     print("Mate in: ",eval.get('value'))
    #     print(eval)

def playGame():
    while stockfish_white.get_best_move():
        # playMove(stockfish_white.get_best_move())
        # os.system('clear')
        # print(stockfish_white.get_board_visual())
        # playMove(stockfish_black.get_best_move())
        # os.system('clear')
        # print(stockfish_white.get_board_visual())

        createDataEntry(whitesTurn=True)
        playMove(stockfish_white.get_best_move_time(100))
        createDataEntry(whitesTurn=False)
        playMove(stockfish_black.get_best_move_time(100))

def playGameLimitedMoves(moves):
    while moves > 0 and stockfish_white.get_best_move() and stockfish_black.get_best_move():
        # playMove(stockfish_white.get_best_move())
        # os.system('clear')
        # print(stockfish_white.get_board_visual())
        # playMove(stockfish_black.get_best_move())
        # os.system('clear')
        # print(stockfish_white.get_board_visual())
        try:
            createDataEntry(whitesTurn=True)
            playMove(stockfish_white.get_best_move_time(100))
            createDataEntry(whitesTurn=False)
            playMove(stockfish_black.get_best_move_time(100))
            moves -= 1
        except:
            print("Couldnt make move")
            moves = 0
    print(stockfish_white.get_board_visual())

def createRandomFen(min_moves=10, max_moves=30):
    board = chess.Board()
    num_moves = random.randint(min_moves, max_moves)
    for i in range(num_moves):
        legal_moves = [move for move in board.legal_moves]
        random_move = random.choice(legal_moves)
        board.push(random_move)
    fen_position = board.fen()
    return fen_position

def setPosition(position):
    stockfish_black.set_fen_position(position)
    stockfish_white.set_fen_position(position)


#playGame()
amount_of_games = 100
while amount_of_games > 0:
    setPosition(createRandomFen(min_moves=100, max_moves=200))
    playGameLimitedMoves(3)
    amount_of_games -= 1
