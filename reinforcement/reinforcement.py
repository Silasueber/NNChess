import os

import chess
import torch
import random
import torch.nn as nn
import torch.optim as optim
from stockfishHelper import initializeStockfish
import numpy as np
import csv
import copy

# Workflow mainly from here: https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# Some GPT conversation: https://chat.openai.com/share/296512a2-3cf9-4378-8bb9-992b3f3c110c


csv_file_name = "data/training.csv"
model_name = "models/reinforcement.pt"
epsilon = 0.95 # for epsilon greedy
gamma = 0.95 # discount factor so immediate rewards are better than later rewards
evaluator = initializeStockfish()

possible_pieces = 5 * 2 + 1 #each player has 5 unique pieces (King, Pawn, Knight, Rook, Bishop) and empty field
fields_in_chess_board = 8*8
input_size = possible_pieces*fields_in_chess_board
no_of_white_pieces = 8 #we play only white, therefore only need white actions
output_size = no_of_white_pieces * fields_in_chess_board # simplified assumptuion that every piece can move everywhere during the game


#2 Workflow --> Generate data
board_viz = initializeStockfish() #TODO different stockfish so i dont mix it up for eval
def create_random_state():
    return chess.Board("2rknb2/2pppp2/8/8/8/8/2PPPP2/2RKNB2 w - - 0 1") # TODO just starting position for now
def setup_environment():
    stockfish = initializeStockfish(1000)
    board = create_random_state()
    return stockfish, board

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
    #TODO, have to have different values than in other files cuz we distinguish between bishop/knight
    piece_values = {'p': 1, 'r': 5, 'n': 4, 'b': 3, 'k': 1000}
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
        if(int(field) == 10):
            print("breakpoint")
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
    if len(square_indices) > 1: #Pawns cuz we have 4 #TODO only in the beginning
        # TODO handle if he wants to move first pawn (c file) but it doesnt exist anymore, so it moves the d file pawn cuz its now in first index
        pawn_index = determine_pawn_from_file(file)
        try:
            current_position_of_piece = chess.square_name(square_indices[pawn_index])
        except:
            return None #piece doesnt exist anymore
    else:
        try:
            current_position_of_piece = chess.square_name(square_indices[0])
        except:
            return None #piece doesnt exist anymore
    #ie with action 3 i expect for pawn1 c2d1
    return current_position_of_piece + file + str(number)


def get_move_from_q_values(state, q_value_action):
    legal = False
    while legal is False:
        # Greedy Epsilon
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, output_size)
        else:
            action_index = np.argmax(q_value_action.detach().numpy())
        move = map_action_indice_to_move(state, action_index)
        try:
            move_algebraic_notation = chess.Move.from_uci(move)
            if move_algebraic_notation in list(state.legal_moves):
                legal = True
                return move_algebraic_notation, action_index
        except:
            legal = False
            #dont handle

def save_example(current_state, action, reward, next_state, action_index, next_state_as_fen):
    if reward is not None:
        current_state_as_csv = ','.join(['%.0f' % num for num in current_state])
        next_state_as_csv = ','.join(['%.0f' % num for num in next_state])

        concatenated_example = current_state_as_csv + "+" + str(action) + "+" + str(reward) + "+" + next_state_as_csv + "+" + str(action_index) + "+" + next_state_as_fen
        try:
            with open(csv_file_name, 'a', newline='') as csv_file:
                csv_file.write(concatenated_example + '\n')
                print(f"save state, {action}, {reward}, next state, action_index, next_state_as_fen to {csv_file_name}")
        except Exception as e:
            print(f"Error in 'save_example': {e}")
    else:
        print("Shouldn't happen: reward was None")

def determine_reward(before_action, after_action):
    # Be careful: the stockfish that evaluats must be always the best possbile version, if stockfish black is not the best change this line
    eval_type_before_action = before_action.get('type')
    eval_type_after_action = after_action.get('type')
    eval_value_before_action = before_action.get("value")
    eval_value_after_action = after_action.get("value")
    change_of_cpawn_value = eval_value_after_action - eval_value_before_action
    if eval_type_before_action == "cp" and eval_type_after_action == "cp":
        # if abs(change_of_cpawn_value) < 50:
        #     return 0
        if change_of_cpawn_value > 0:
            return +1
        elif change_of_cpawn_value < 0:
            return -1
        else:
            return 0
    elif eval_type_before_action == "cp" and eval_type_after_action == "mate":
        if eval_value_after_action > 0:
            return 10
        elif eval_value_after_action < 0:
            return -10 # means we did a bad action to set ourselves mate in x turns
        else:
            return 0
    elif eval_type_before_action == "mate" and eval_type_after_action == "mate":
        if change_of_cpawn_value > 0:
            return +1
        elif change_of_cpawn_value < 0:
            return -1
        else:
            return 0
    elif eval_type_before_action == "mate" and eval_type_after_action == "cp":
        if eval_value_before_action > 0:
            return -5 #had a mate and didnt do the move
        elif eval_value_before_action < 0:
            return +1 # was in mate and got better situation

def create_new_example(state, enemy_player, q_net):
    #transform state into one hot
    current_state = transformSingleBoardToOneHot(state)
    # evaluate current position
    evaluator.set_fen_position(state.fen())
    print("not stuck until evaluator.set_fen_position(state.fen())")
    before_action_eval = evaluator.get_evaluation()
    print("not stuck until before_action_eval = evaluator.get_evaluation()")
    input_for_net = torch.tensor(current_state, dtype=torch.float32)
    print("not stuck until input_for_net = torch.tensor(current_state, dtype=torch.float32)")
    # put state into NN
    q_values = q_net(input_for_net)
    print("not stuck until  q_values = q_net(input_for_net)")

    # greedy epsilon with output and check if its legal, otherwise go do it again
    agent_move, action_index = get_move_from_q_values(state, q_values)
    print("not stuck until agent_move, action_index = get_move_from_q_values(state, q_values)")
    # do step
    state.push(agent_move)
    print("not stuck until state.push(agent_move)")
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
    save_example(current_state, agent_move, reward, next_state, action_index, state.fen())
    return state

def create_new_examples(games, turns, q_net):
    for game in range(games):
        enemy_player, board = setup_environment()
        next_state = board
        print("Starting new game")
        for turn in range(turns):
            next_state = create_new_example(next_state, enemy_player, q_net)
            if next_state.is_game_over():
                print("Game over!")
                break

def get_number_of_rows_in_training_set():
    with open(csv_file_name) as f:
       return sum(1 for row in f)

def transform_csv_row_into_parts(row):
    row = ",".join(row)
    parts = row.split("+")
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
def load_training_data(batch_indices):
    current_states = np.array([])
    actions = np.array([])
    rewards = np.array([])
    next_states = np.array([])
    action_indices = np.array([], dtype=int)
    next_states_as_fen = np.array([])
    with open(csv_file_name) as f:
        reader = csv.reader(f)
        training_rows = [row for idx, row in enumerate(reader) if idx in batch_indices]
    for row in training_rows:
        current_state, action, reward, next_state, action_index, next_state_as_fen = transform_csv_row_into_parts(row)
        current_states = np.append(current_states, np.fromstring(current_state, sep=",", dtype=float))
        actions = np.append(actions, action)
        rewards = np.append(rewards, int(reward))
        next_states = np.append(next_states, np.fromstring(next_state, sep=",", dtype=float))
        action_indices = np.append(action_indices, int(action_index))
        next_states_as_fen = np.append(next_state_as_fen, int())
    return current_states.reshape(len(training_rows), input_size),\
        actions,\
        rewards,\
        next_states.reshape(len(training_rows), input_size),\
        action_indices

def load_model():
    if os.path.isfile(model_name):
        return torch.load(model_name)
    else:
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
def get_q_value_of_selected_action(q_values, action_indices):
    # Map action to index of q_values
    selected_q_values = torch.empty(q_values.shape[0])
    for i in range(q_values.shape[0]):
        selected_q_values[i] = q_values[i][action_indices[i]]

    return selected_q_values
def uci_to_algebraic_notation(uci):
    try:
        return chess.Move.from_uci(uci)
    except:
        return chess.Move.from_uci("0000")
def get_highest_legal_q_value_from_predictions(state, q_values):
    found_legal_move = False
    try:
        while not found_legal_move and q_values.shape[0] > 0:
            index_of_action = int(torch.argmax(q_values))
            move = map_action_indice_to_move(state, index_of_action)
            if move is not None:
                move_algebraic_notation = uci_to_algebraic_notation(move)
                if move_algebraic_notation in list(state.legal_moves):
                    return q_values[index_of_action], index_of_action
                else:  # else we remove this element from tensor cause its an illegal move
                    q_values = torch.cat(
                        [q_values[0:index_of_action], q_values[index_of_action + 1:]])
            else:  # else we remove this element from tensor cause its an illegal move
                q_values = torch.cat(
                    [q_values[0:index_of_action], q_values[index_of_action + 1:]])
        return None, None
    except:
        print("why")
def select_best_values_for_each_example(predicted_target_values, next_states_as_fen):

    max_prediction_values = torch.empty(predicted_target_values.shape[0])
    for i in range(predicted_target_values.shape[0]):
        considered_state_as_fen = next_states_as_fen[i]
        considered_tensor = predicted_target_values[i]
        max_prediction_values[i], _ = get_highest_legal_q_value_from_predictions(considered_state_as_fen, considered_tensor)

        # while not found_legal_move and considered_tensor.shape[0] > 0:
        #     index_of_action = torch.argmax(considered_tensor)
        #     move = map_action_indice_to_move(considered_state_as_fen, index_of_action)
        #     move_algebraic_notation = chess.Move.from_uci(move)
        #     if move_algebraic_notation in list(considered_state_as_fen.legal_moves):
        #         max_prediction_values[i] = considered_tensor[index_of_action]#torch.max(considered_tensor)
        #         found_legal_move = True
        #     else: # else we remove this element from tensor cause its an illegal move
        #         considered_tensor = torch.cat([considered_tensor[0:index_of_action], considered_tensor[index_of_action+1:]])
    return max_prediction_values
def train(epochs, batch_size):
    #load model if exists
    q_net = load_model()
    target_net = copy.deepcopy(q_net) # otherwise same object
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(q_net.parameters(), lr=0.01)
    for epoch in range(epochs):
        #every 5 epochs lower the epsilon value
        if epoch % 5 == 0:
            global epsilon
            epsilon = epsilon * 0.95
        #new training data
        #create_new_examples(1, 20, q_net)
        number_of_rows = get_number_of_rows_in_training_set()
        possible_indices = [*range(0, number_of_rows, 1)]
        batch_indices = random.sample(possible_indices, batch_size)
        #load random batch of training data
        current_states, actions, rewards, next_states, action_indices, next_states_as_fen = load_training_data(batch_indices)
        X_qnet = torch.tensor(current_states, dtype=torch.float32)
        X_tnet = torch.tensor(next_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # q-net predicts value ONLY for the one action we actually selected in the data creation
        predicted_q_values = q_net(X_qnet)
        value_of_selected_actions = get_q_value_of_selected_action(predicted_q_values, action_indices)

        # target predicts target q value
        predicted_target_values = target_net(X_tnet)
        # best legal action that can be taken from these states (target q value)
        best_q_values = select_best_values_for_each_example(predicted_target_values, next_states_as_fen)
        # Target Q Value is output of target plus reward from sample
        target_q_values = rewards_tensor + gamma * best_q_values
        # compute loss
        loss = loss_fn(value_of_selected_actions, target_q_values)
        # train q network, target net is fixed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

        if epoch % 10 == 0:
            target_net = copy.deepcopy(q_net)
    # save model
    torch.save(q_net, model_name)
#train(1, 600)
#create_new_examples(50, 20, load_model())


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