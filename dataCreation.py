import chess
import csv
import random
from stockfishHelper import initalizeStockfish

# Init Stockfish parameters
stockfish_white = initalizeStockfish()
stockfish_black = initalizeStockfish()
#stockfish_white.set_elo_rating(1000)
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



def createDataEntry(whitesTurn):
    # Dataset one with 1 -> White winning 0 -> Black winning 0.5 -> Draw
    csv_path = "data/p1.csv"
    position = convertPositionToString(stockfish_black.get_board_visual())
    eval = convertEvalIntoValue()
    turn = "1," if whitesTurn else "0,"
    line = turn+position+","+str(eval)
    try:
        with open(csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(line.split(','))
    except Exception as e:
        print(f"Error: {e}")

    # Dataset two with [1,0,0] -> White winning [0,1,0] -> Black winning [0,0,1] -> Draw
    csv_path = "data/p2.csv"
    winner = ""
    if eval == 1:
        winner = "1,0,0"
    elif eval == 0: 
        winner = "0,1,0"
    else:
        winner = "0,0,1"

    line = turn+position+","+winner
    try:
        with open(csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(line.split(','))
    except Exception as e:
        print(f"Error: {e}")

    # Dataset three with cpawn value for position [limited at -10 and +10]
    csv_path = "data/p3_2.csv"
    winner = getCpawnValue()
    line = turn+position+","+str(winner)
    try:
        with open(csv_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(line.split(','))
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
            return 0
    else:
        if eval_value > 100: 
            # Winning white
            return 1
        elif eval_value < -100: 
            # Winning Black
            return 0
        else:
            # Equal
            return 0.5
    
def getCpawnValue():
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    eval = stockfish_black.get_evaluation()
    eval_type = eval.get('type')
    eval_value = round((eval.get("value") + 700)/1400*0.8+0.1,4)
    
    if eval_value > 0.9:
        eval_value = 0.9
    if eval_value < 0.1:
        eval_value = 0.1
    if eval_type == "mate":
        if eval.get('value') > 0: 
            return round(0.9+(0.1/eval.get("value")),4)
        else: 
            # Winning Black
            return round(0.1-(0.1/(-eval.get("value"))),4)
    else:
        return eval_value

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
        
        # if getCpawnValue() == 0.5: break # Dont flood Database with dead draws
        try:
            # PLAY BEST MOVE

            createDataEntry(whitesTurn=True)
            playMove(stockfish_white.get_best_move_time(100))
            createDataEntry(whitesTurn=False)
            playMove(stockfish_black.get_best_move_time(100))

            moves -= 1
        except Exception as e:
            print(e)
            moves = 0
    #print(stockfish_white.get_board_visual())

def createRandomFen(min_moves=10, max_moves=30):
    board = chess.Board()
    num_moves = random.randint(min_moves, max_moves)
    for i in range(num_moves):
        if board.is_game_over(): break
        try:
            legal_moves = [move for move in board.legal_moves]
            random_move = random.choice(legal_moves)
            board.push(random_move)
        except: 
            print("Couldnt make move")
    fen_position = board.fen()
    return fen_position

def setPosition(position):
    stockfish_black.set_fen_position(position)
    print(stockfish_black.get_board_visual())
    stockfish_white.set_fen_position(position)


if __name__ == "__main__":
    amount_of_games = int(input("Amount of Games: "))
    custom_position = input("Costum Position (if null random fen gets generated): ")
    if not stockfish_black.is_fen_valid(custom_position):
        print("Invalid Fen position")
        min_moves = int(input("Minimum Moves to generate random Position: "))
        max_moves = int(input("Maximum Moves to generate random Position: "))
        moves_to_play = int(input("Moves per Position to play: "))
        for i in range(amount_of_games):
            print("Current Game: " + str(i+1))
            setPosition(createRandomFen(min_moves=min_moves, max_moves=max_moves))
            playGameLimitedMoves(moves_to_play)
    else: 
        moves_to_play = int(input("Moves per Position to play: "))
        for i in range(amount_of_games):
            print("Current Game: " + str(i+1))
            setPosition(custom_position)
            playGameLimitedMoves(moves_to_play)

