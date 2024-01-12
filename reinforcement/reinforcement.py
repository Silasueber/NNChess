import chess
import torch
import torch.nn as nn
from stockfishHelper import initializeStockfish
import numpy as np
import csv

# Workflow mainly from here: https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# Some GPT conversation: https://chat.openai.com/share/296512a2-3cf9-4378-8bb9-992b3f3c110c


csv_file_name = "data/training.csv"
# Workflow 1.:
possible_pieces = 6 * 2 + 1 #each player has 6 unique pieces (King, Queen, Pawn, Knight, Rook, Bishop) and empty field
fields_in_chess_board = 8*8
input_size = possible_pieces*fields_in_chess_board
no_of_white_pieces = 14 #we play only white, therefore only need white actions
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
    return chess.Board()  # TODO just random board for now
def setup_environment():
    stockfish = initializeStockfish(1000)
    board = create_random_state()
    evaluator = initializeStockfish()
    return stockfish, board, evaluator


#TODO put into own file and dont duplicate
one_hot_mapping = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # White Pawn
    3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # White Bishop
    4: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # White Knight
    5: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # White Rook
    10: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # White Queen
    1000: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # White King
    -1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Black Pawn
    -3: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Black Bishop
    -4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Black Knight
    -5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Black Rook
    -10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Black Queen
    -1000: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # Black King
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

# def determine_turn(state):
#     if state.turn == chess.WHITE:
#         return 1
#     else
#         return 0
def transformSingleBoardToOneHot(state_param):

    # players_turn = determine_turn(state) # TODO bruacht man wohl nicht? Weil wir immer assumen dass wir weiß sind und current state und der next state immer wir dran sind
    board_viz.set_fen_position(state_param.fen())
    state_param = convertPositionToString(board_viz.get_fen_position())
    state_param = state_param.split(',')
    newBoardRepresentation = np.array([])
    for field in state_param[:]:
        newBoardRepresentation = np.append(newBoardRepresentation, one_hot_mapping[int(field)])

    return newBoardRepresentation



# def createDataEntry(whitesTurn):
#     position = convertPositionToString(stockfish_black.get_board_visual())
#     line = turn+position+","+str(eval)
#     try:
#         with open(csv_file_name, 'a', newline='') as csv_file:
#             csv_writer = csv.writer(csv_file)
#             csv_writer.writerow(line.split(','))
#     except Exception as e:
#         print(f"Error: {e}")

def get_move_from_output(actions):
    # TODO how to map the output back to an applicable action?
    # simplification might be only being able to move one piece
    # select move with greedy epsilon
    # check if move is legal
        # if not --> do greedy-epsilon again
    #return move
    return None

def save_example(current_state, action, reward, next_state):
    #can split by '+' to get all previous parts again
    concatatenated_example = str(current_state) + "+"+ str(action) + "+" + str(reward) + "+" + str(next_state)
    try:
        with open(csv_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(concatatenated_example)
    except Exception as e:
        print(f"Error in 'save_example': {e}")

def create_new_example(state):
    #transform state into one hot
    current_state = transformSingleBoardToOneHot(state)
    # evaluate current position
    evaluator.set_fen_position(state.fen())
    before_action_eval = evaluator.get_evaluation()
    input_for_net = torch.tensor(current_state, dtype=torch.float32)
    # put state into NN
    actions = q_net(input_for_net)
    # greedy epsilon with output and check if its legal, otherwise go do it again
    agent_move = get_move_from_output(actions)
    # do step
    state.push(agent_move)
    # calculate reward
    evaluator.set_fen_position(state.fen())
    after_action_eval = evaluator.get_evaluation()
    reward = 0 #TODO
    # compare before_action_eval and after_action_eval
    # do step with enemy
    best_enemy_move = enemy_player.get_best_move_time(200)
    state.push(best_enemy_move)
    next_state = state #TODO state after enemy moved
    #save as example
    save_example(current_state, agent_move, reward, next_state )
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