import chess
import torch
import torch.nn as nn
from stockfishHelper import initializeStockfish
import numpy as np
import csv

# Workflow mainly from here: https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# Some GPT conversation: https://chat.openai.com/share/296512a2-3cf9-4378-8bb9-992b3f3c110c


csv_file_name = "data/training.csv"
epsilon = 0.95
# Workflow 1.:
possible_pieces = 5 * 2 + 1 #each player has 5 unique pieces (King, Pawn, Knight, Rook, Bishop) and empty field
fields_in_chess_board = 8*8
input_size = possible_pieces*fields_in_chess_board
no_of_white_pieces = 8 #we play only white, therefore only need white actions
output_size = no_of_white_pieces * fields_in_chess_board # simplified assumptuion that every piece can move everywhere during the game
q_net = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(256, 32),
    nn.ReLU(),
    nn.Linear(32, output_size)
)
target_net = q_net # copy q_net to target net

#2 Workflow --> Generate data
board_viz = initializeStockfish() #TODO different stockfish so i dont mix it up for eval
# See how much functionality we need, otherwise put in own class/file
def create_random_state():
    return chess.Board("2rknb2/2pppp2/8/8/8/8/2PPPP2/2RKNB2 w - - 0 1")  # TODO just starting board for now
def setup_environment():
    stockfish = initializeStockfish(1000)
    board = create_random_state()
    evaluator = initializeStockfish() #must be best version of stockfish i think
    return stockfish, board, evaluator


#TODO put into own file and dont duplicate
one_hot_mapping = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],       # Empty
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],       # White Pawn
    3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],       # White Bishop
    4: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],       # White Knight
    5: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],       # White Rook
    1000: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],    # White King
    -1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],      # Black Pawn
    -3: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],      # Black Bishop
    -4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],      # Black Knight
    -5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],      # Black Rook
    -1000: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # Black King
}


def convertPositionToString(fen):
    #TODO, have to have different values
    piece_values = {'p': 1, 'r': 5, 'n': 4, 'b': 3, 'q': 10, 'k': 1000}
    fen_board = chess.Board(fen)
    fen_board = str(fen_board)
    lines = fen_board.split('\n')

    result = []
    for line in lines:
        for char in line.split(' '):
            char = char.strip()
            if char.lower() in piece_values:
                value = piece_values[char.lower()]
                result.append(str(value) if char.islower() else str(-value))
            else:
                result.append('0')

    return ','.join(result)

def transformSingleBoardToOneHot(state_param):
    board_viz.set_fen_position(state_param.fen())
    state_param = convertPositionToString(board_viz.get_fen_position())
    state_param = state_param.split(',')
    newBoardRepresentation = np.array([])
    for field in state_param[:]:
        newBoardRepresentation = np.append(newBoardRepresentation, one_hot_mapping[int(field)])

    return newBoardRepresentation

def get_piece_type_of(piece_name):
    match piece_name:
        case "Rook":
            return chess.ROOK
        case "King":
            return chess.KING
        case "Knight":
            return chess.KNIGHT
        case "Bishop":
            return chess.BISHOP
        case _:
            return chess.PAWN


def determine_pawn_from_file(file):
    match file:
        case "c":
            return 0
        case "d":
            return 1
        case "e":
            return 2
        case "f":
            return 3
        case _:
            return 0 # cant move there anways cuz its not on a pawn file so we dont care about it. just so we dont get errors



def map_action_indice_to_move(state, action):
    piece_map = ["Pawn1", "Pawn2", "Pawn3", "Pawn4",
                "Rook", "King", "Knight", "Bishop"]
    piece_name = piece_map[action // 64]
    file = chr(action % 8 + 97)
    number = (action % 64) // 8 + 1
    piece_type = get_piece_type_of(piece_name)
    square_indices = list(state.pieces(piece_type, chess.WHITE)) #fields on board are numbered from 0 to 63
    if len(square_indices) > 1: #Pawns cuz we have 4
        pawn_index = determine_pawn_from_file(file)
        current_position_of_piece = chess.square_name(square_indices[pawn_index])
    else:
        current_position_of_piece = chess.square_name(square_indices[0])
    #ie with action 3 i expect for pawn1 c2d1
    return current_position_of_piece + file + str(number)


def get_move_from_q_values(state, q_value_action):

    legal = False
    while legal is False:
        # Greedy Epsilon TODO variable with time
        if np.random.rand() < epsilon:
            action = np.random.randint(0, output_size)
        else:
            action = np.argmax(q_value_action.detach().numpy())
        move = map_action_indice_to_move(state, action)
        try:
            move_algebraic_notation = chess.Move.from_uci(move)
            if move_algebraic_notation in list(state.legal_moves):
                legal = True
                return move_algebraic_notation
        except:
            legal = False
            #dont handle

def save_example(current_state, action, reward, next_state):
    current_state_as_csv = ','.join(['%.0f' % num for num in current_state])
    next_state_as_csv = ','.join(['%.0f' % num for num in next_state])

    concatenated_example = current_state_as_csv + "+"+ str(action) + "+" + str(reward) + "+" + next_state_as_csv
    try:
        with open(csv_file_name, 'a', newline='') as csv_file:
            csv_file.write(concatenated_example + '\n')
    except Exception as e:
        print(f"Error in 'save_example': {e}")

def determine_reward(before_action, after_action):
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    eval_type_before_action = before_action.get('type')
    eval_type_after_action = after_action.get('type')
    eval_value_before_action = before_action.get("value")
    eval_value_after_action = after_action.get("value")
    # TODO can also be other type right? == "mate"
    if eval_type_before_action == "cp" and eval_type_after_action == "cp":
        return eval_value_after_action - eval_value_before_action


def create_new_example(state):
    #transform state into one hot
    current_state = transformSingleBoardToOneHot(state)
    # evaluate current position
    evaluator.set_fen_position(state.fen())
    before_action_eval = evaluator.get_evaluation()
    input_for_net = torch.tensor(current_state, dtype=torch.float32)
    # put state into NN
    q_values = q_net(input_for_net)
    # greedy epsilon with output and check if its legal, otherwise go do it again
    agent_move = get_move_from_q_values(state, q_values)
    # do step
    state.push(agent_move)
    # calculate reward
    evaluator.set_fen_position(state.fen())
    after_action_eval = evaluator.get_evaluation()
    reward = determine_reward(before_action_eval, after_action_eval)
    # do step with enemy
    enemy_player.set_fen_position(state.fen())
    best_enemy_move = enemy_player.get_best_move_time(200)
    state.push(chess.Move.from_uci(best_enemy_move))
    next_state = transformSingleBoardToOneHot(state)
    #save as example
    save_example(current_state, agent_move, reward, next_state)
    print(f"save state, action, reward, next state to {csv_file_name}")

enemy_player, board, evaluator = setup_environment()
create_new_example(board)

### Workflow
# 1. Init Q network with random weights and copy to target net
#   --> How many input/output neurons?
#   --> How many and which layers?
# 2. Generate data with experience replay (select action with epsilon-greedy)
# 2.1 --> Q net acts as agent to interact with env
# 2.2 --> Q net predicts q values of all actions
# 2.3 --> Use that prediction to select epsilon greedy action
# 2.4 --> Sample data (current sate, action, reward, next state) is saved
# 3. Train
# 3.1 Select random batch from replay data as input for q and target net
# 3.2 Predict q value for current training example (input is state and action)
# 3.3 Target net predicts Q values for each action and selects maximum of those
# 3.3.1 Target Q value is Target net output + reward from sample
# 3.4 Compute loss
# 3.4.1 MSE between Target Q Value and Predicted Q Value
# 3.5 Backprop/Gradient Descent
#       --> Target net remains fixed (no training)

### (Open) Questions
# 1: Representing states
#   Possiblities:
#   --> One Hot of each Field --> Fixed size input (makes sense says GPT)
#       --> King, Queen, Pawn, Rook, Knight, 2xBishop --> 14 different pieces (black/white) and empty space
#            --> vector is of size 15 (GPT says use 16 cuz NN can learn better this way)
#               --> Input is 8*8*15 = 960 cuz field is 8x8 and each field has one hot vector of 15
#   --> Use CNN to capture spatial information
# 2: Representing actions
#   --> Actions are output of network and we have A LOT
#       --> For each possible field each possible action
#       --> Naive modeling: We have 64 outputs for every piece
#           --> I imagine it being like "Move pawn to e5" and it doesnt matter where it is right now"
#               --> Have to deal with invalid action (ie choose smth and then see if legal, otherwise choose again)
#           --> We only consider one side, therefore 16 pieces*64 fields = 1024 outputs
#            --> OR MINI CHESS (2rnkr2/2pppp2/8/8/8/8/2PPPP2/2RNKR2 w - - 0 1)
#   --> Limit?
# 3: Choice of reward
#  --> When will the reward be calculated and on what basis?
#  --> Reward could be current win probability - win probability after action (but before move of other player)
#       --> This approach is most simple probably
#  --> Reward could be current win probability - win probability of next state (so after enemy player moved)
#  --> Lose/win of pieces. Ie "killing a pawn" is +1, losing one is -1, with cpawn value stuffz
# 4: Who is going to be the enemy?
#   --> Stockfish max too stronk i guess
#   --> Random enemy moves dont punish us and our gottlosen blunders
#   --> Beginner level ELO best? This way they get punished for obvious blunders and can still learn a bit?