import argparse
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

#Read hyperparameters and configuration parameters
parser = argparse.ArgumentParser()
parser.add_argument("--mode", nargs="?",
                    help="Choose to create examples (--mode=examples) with 10 games and 20 turns each "
                         "or train the model (--mode=train) (default: None)")
parser.add_argument("--epochs", nargs="?",
                    help="Number of epochs to train (default: 10)")
parser.add_argument("--batch", nargs="?",
                    help="Batch size of training examples used for each epoch (default: 600)")
parser.add_argument("--name", nargs="?",
                    help="Name to load/save model (default: reinforcement.pt)")
parser.add_argument("--dataset", nargs="?",
                    help="Dataset used for training and saving new examples when interacting with the environment (default: training.csv")
parser.add_argument("--lr", nargs="?",
                    help="Learning rate for the model (default: 0.001)")
parser.add_argument("--epsilon", nargs="?",
                    help="Epsilon for epsilon greedy (default: 0.95)")
parser.add_argument("--gamma", nargs="?",
                    help="Gamma to discount future rewards (default: 0.95)")
parser.add_argument("--enemy-elo", nargs="?",
                    help="Elo of the Stockfish enemy player in the environment (default: 1000)")
args, unknown = parser.parse_known_args()
evaluator = initializeStockfish() #Best Stockfish possible to make evaluations

# Initialization of parameters at end of file below all methods
# Initial environment setup
#csv_file_name = "data/training.csv"
#epsilon = 0.95 # for epsilon greedy
#gamma = 0.95 # discount factor so immediate rewards are better than later rewards
#enemy_player = initializeStockfish(1000)

#Explanation of network input and output sizes
possible_pieces = 5 * 2 + 1 #each player has 5 unique pieces (King, Pawn, Knight, Rook, Bishop) and empty field
fields_in_chess_board = 8*8 # 1-8 and a-h
input_size = possible_pieces*fields_in_chess_board # so each field can have 11 unique pieces on it
no_of_white_pieces = 8 #we play only white, therefore only need white actions
output_size = no_of_white_pieces * fields_in_chess_board # simplified assumptuion that every piece can move everywhere during the game
#One Hot Encoding of the chess squares to act as input for the Neural Network
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
def create_starting_board():
    """
    Returns a new mini-chess board in starting position
    :return: New chess board
    """
    return chess.Board("2rknb2/2pppp2/8/8/8/8/2PPPP2/2RKNB2 w - - 0 1")




def convertPositionToString(fen):
    """
    Maps a FEN String to a numerical representation: {'p': 1, 'r': 5, 'n': 4, 'b': 3, 'k': 1000}
    :param fen: A FEN String
    :return: String representation of the board that is
    """
    # We dont consider the queen because its mini-chess
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
    """
    Transforms a chess.Board into a flat one hot encoded vector
    :param state_param: board
    :return: Flat one hot encoding of the board
    """
    evaluator.set_fen_position(state_param.fen())
    state_param = convertPositionToString(evaluator.get_fen_position())
    state_param = state_param.split(',')
    newBoardRepresentation = np.array([])
    for field in state_param[:]:
        newBoardRepresentation = np.append(newBoardRepresentation, one_hot_mapping[int(field)])

    return newBoardRepresentation

def get_piece_type_of(piece_name):
    """
    Maps the piece name to the chess-library enum
    :param piece_name: The name of the piece to be mapped
    :return: The mapping from the chess library to the piece_name
    """
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


def determine_pawn_from_file(file): #TODO only works if there are all starting pawns alive
    """
    Maps the file to the associated pawn
    :param file: The file of the board
    :return: Index of the pawn from the piece_map
    """
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
    """
    Maps the action index to a move that the chess engine can understand
    :param state: Current state
    :param action: Considered action as an index
    :return: The UCI representation of a move or None if the associated piece doesn't exist anymore
    """
    if action is None:
        return None
    piece_map = ["Pawn1", "Pawn2", "Pawn3", "Pawn4",
                "Rook", "King", "Knight", "Bishop"]
    piece_name = piece_map[action // 64]
    file = chr(action % 8 + 97)
    number = (action % 64) // 8 + 1
    piece_type = get_piece_type_of(piece_name)
    square_indices = list(state.pieces(piece_type, chess.WHITE))
    # Can have more than one square_indices if there are multiple pawns
    if len(square_indices) > 1:
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


def get_epsilon_greedy_move_from_q_values(state, q_value_action):
    """
    Using Epsilon-Greedy it chooses a move for the agent to play
    :param state: Considered board
    :param q_value_action: The predicted Q-Values from the Network
    :return: A legal move and its index or (chess.Move.from_uci("0000"), -1) if too many iterations were needed to find a legal move
    """
    number_of_tries_to_find_legal_move = 0

    while True:
        #otherwise we might be taking forever to get a legal move if we have to find a single one by chance
        if number_of_tries_to_find_legal_move > 512:
            return chess.Move.from_uci("0000"), -1
        # Greedy Epsilon
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, output_size)
        else:
            action_index = np.argmax(q_value_action.detach().numpy())
        move = map_action_indice_to_move(state, action_index)
        move_algebraic_notation = uci_to_algebraic_notation(move)
        if move_algebraic_notation in list(state.legal_moves):
            return move_algebraic_notation, action_index
        number_of_tries_to_find_legal_move += 1


def save_example(current_state, action, reward, next_state, action_index, next_state_as_fen):
    """
    Saves the current example in a csv format
    :param current_state:
    :param action:
    :param reward:
    :param next_state:
    :param action_index:
    :param next_state_as_fen:
    :return:
    """
    if reward is not None:
        current_state_as_csv = ','.join(['%.0f' % num for num in current_state])
        next_state_as_csv = ','.join(['%.0f' % num for num in next_state])

        concatenated_example = current_state_as_csv + "+" + str(action) + "+" + str(reward) + "+" + next_state_as_csv + "+" + str(action_index) + "+" + next_state_as_fen
        try:
            with open(csv_file_name, 'a', newline='') as csv_file:
                if action_index != -1:
                    csv_file.write(concatenated_example + '\n')
                    print(f"save state, {action}, {reward}, next state, action_index, next_state_as_fen to {csv_file_name}")
                else:
                    "didnt save state because action_index was invalid"
        except Exception as e:
            print(f"Error in 'save_example': {e}")
    else:
        print("Shouldn't happen: reward was None")

def determine_reward(before_action, after_action):
    """
    Calculating the reward using the cpawn/mate value of the Stockfish-Engine
    :param before_action: cpawn/mate value before agent executed action
    :param after_action: cpawn/mate value after agent executed action
    :return: Reward for the agent
    """
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
    """
    Creates a new example from a state, an enemy and the current network
    :param state: Current state
    :param enemy_player: Enemey player
    :param q_net: Our Q-Network
    :return: The state after the agent and enemy player each did a turn
    """
    #transform state into one hot
    current_state = transformSingleBoardToOneHot(state)
    # evaluate current position
    evaluator.set_fen_position(state.fen())
    before_action_eval = evaluator.get_evaluation()
    input_for_net = torch.tensor(current_state, dtype=torch.float32)
    # put state into NN
    q_values = q_net(input_for_net)
    # greedy epsilon with output and check if its legal, otherwise go do it again
    agent_move, action_index = get_epsilon_greedy_move_from_q_values(state, q_values)
    if action_index == -1:
        return None
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
    save_example(current_state, agent_move, reward, next_state, action_index, state.fen())
    return state

def create_new_examples(games, turns, q_net):
    """
    Creates new examples to serve as training data
    :param games: Amount of games to be played
    :param turns: Amount of turns in each game
    :param q_net: The Network to be used for selectin agent actions
    :return:
    """
    for game in range(games):
        board = create_starting_board()
        next_state = board
        print("Starting new game")
        for turn in range(turns):
            next_state = create_new_example(next_state, enemy_player, q_net)
            if next_state is None or next_state.is_game_over():
                print("Game over or no action could be found in reasonable time!")
                break

def get_number_of_rows_in_training_set():
    """
    Determines the number of rows in the training set
    :return: NUmber of rows in training set
    """
    with open(csv_file_name) as f:
       return sum(1 for row in f)

def transform_csv_row_into_parts(row):
    """
    Splits the row up into: current_state, action, reward, next_state, action_index, next_state_as_fen
    :param row: Row to  be split
    :return: current_state, action, reward, next_state, action_index, next_state_as_fen
    """
    row = ",".join(row)
    parts = row.split("+")
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
def load_training_data(batch_indices):
    """
    Loads each row of the training data already split up into the columns
    :param batch_indices: Training examples to be loaded
    :return: Lists of each column from the training data
    """
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
        next_states_as_fen = np.append(next_states_as_fen, next_state_as_fen)
    return current_states.reshape(len(training_rows), input_size),\
        actions,\
        rewards,\
        next_states.reshape(len(training_rows), input_size),\
        action_indices, \
        next_states_as_fen

def load_model(model_name):
    """
    Loads the model if it exists, otherwise creates new one
    :param model_name: Name of the model
    :return: The model
    """
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
def get_q_value_of_selected_actions(q_values, action_indices):
    """
    Returns the Q-Values of the selected actions
    :param q_values: List of Q-Values predicted by a network
    :param action_indices: List of indices of the actions taken
    :return: List of predicted Q-Value of the selected action in the order the list of Q-Value-Predictions were provided
    """
    # Map action to index of q_values
    selected_q_values = torch.empty(q_values.shape[0])
    for i in range(q_values.shape[0]):
        selected_q_values[i] = q_values[i][action_indices[i]]

    return selected_q_values
def uci_to_algebraic_notation(uci):
    """
    Maps the UCI move to the corresponding algebraic notation
    :param uci: UCI String to convert
    :return: Algebraic notation of the UCI move or a non-move if the UCI move (such as "a1a1") was illegal
    """
    try:
        return chess.Move.from_uci(uci)
    except:
        # if illegal move such as "a1a1" happens we just return "0000" to signal that no move is being done
        return chess.Move.from_uci("0000")

def get_highest_legal_q_value_from_predictions(state, q_values):
    """
    Returns the highest legal Q-Value from the given Q-Values and ensures that it's a legal move
    :param state: Current state of board
    :param q_values: Q-Values predicted by network
    :return: The legal move with the highest Q-Value
    """
    found_legal_move = False
    # copy tensor to another one which we can shorten
    copy_of_q_values = copy.copy(q_values)
    # check if highest move is legal
    while found_legal_move is False and copy_of_q_values.shape[0] > 0:
        highest_q_value = torch.max(copy_of_q_values)
        index_of_highest_q_value_in_copy = torch.argmax(copy_of_q_values)
        #Match element in original q_values and return the index of that element
        index_in_orginal_q_values = ((q_values == highest_q_value).nonzero(as_tuple=True)[0])
        #if theres more than 1 element matching, we just take the first one
        if index_in_orginal_q_values.shape[0] > 1:
            index_in_orginal_q_values = index_in_orginal_q_values[0]
        move = map_action_indice_to_move(state, int(index_in_orginal_q_values))
        move = uci_to_algebraic_notation(move)
        if move is not None and move in list(state.legal_moves):
            found_legal_move = True
            return move
        else:
            copy_of_q_values = torch.cat(
                [copy_of_q_values[0:index_of_highest_q_value_in_copy], copy_of_q_values[index_of_highest_q_value_in_copy + 1:]])

def select_best_values_for_each_example(predicted_target_values, next_states_as_fen):
    """
    For each of the lists of Q-Values (predicted_target_values) it returns the highest Q-Value legal move for each
    :param predicted_target_values: List of Q-Value tensor
    :param next_states_as_fen: List of board states in FEN representation
    :return: List of highest legal move Q-Values for each provided tensor of Q-Values
    """
    max_prediction_values = torch.empty(predicted_target_values.shape[0])
    for i in range(predicted_target_values.shape[0]):
        considered_state_as_fen = next_states_as_fen[i]
        state_of_considered_board = chess.Board(considered_state_as_fen)
        considered_tensor = copy.copy(predicted_target_values[i])
        found_legal_move = False
        while not found_legal_move and considered_tensor.shape[0] > 0:
            highest_q_value = torch.max(considered_tensor)
            index_of_highest_q_value_in_copy = torch.argmax(considered_tensor)
            index_in_orginal_q_values = ((predicted_target_values[i] == highest_q_value).nonzero(as_tuple=True)[0])
            if index_in_orginal_q_values.shape[0] > 1: #if multiple elements have same value
                index_in_orginal_q_values = index_in_orginal_q_values[0]
            move = map_action_indice_to_move(state_of_considered_board, int(index_in_orginal_q_values))
            move = uci_to_algebraic_notation(move)
            if move is not None and move in list(state_of_considered_board.legal_moves):
                found_legal_move = True
                max_prediction_values[i] = highest_q_value
            else: # else we remove this element from tensor cause its an illegal move
                considered_tensor = torch.cat([considered_tensor[0:index_of_highest_q_value_in_copy], considered_tensor[index_of_highest_q_value_in_copy + 1:]])
    return max_prediction_values
def train(epochs, batch_size, lr):
    """
    Trains the network with a number of epochs and batch-size and saves it
    :param epochs: Number of epochs to be processed
    :param batch_size: Size of batch used for each epoch
    :return:
    """
    #load model if exists
    q_net = load_model(model_name)
    target_net = copy.deepcopy(q_net) # otherwise it'd be the same object
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    for epoch in range(epochs):
        #every 5 epochs lower the epsilon value
        if epoch % 5 == 0:
            global epsilon
            epsilon = epsilon * 0.95
        #new training data each epoch with one game and 20 turns
        create_new_examples(1, 20, q_net)
        print("Training the model...")
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
        value_of_selected_actions = get_q_value_of_selected_actions(predicted_q_values, action_indices)

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
        # TODO uncomment sometime
        # if epoch % 10 == 0:
        #     target_net = copy.deepcopy(q_net)
    # save model
    torch.save(q_net, model_name)


#Set arguments passed in call
if args.epochs is not None:
    n_epochs = int(args.epochs)
else:
    n_epochs = 10

if args.batch is not None:
    batch_size = int(args.batch)
else:
    batch_size = 600


if args.dataset is not None:
    csv_file_name = f"data/{args.mode}"
else:
    csv_file_name = "data/training.csv"

if args.name is not None:
    model_name = f"models/{args.name}"
else:
    model_name = "models/reinforcement.pt"

if args.dataset is not None:
    csv_file_name = f"data/{args.dataset}.csv"
else:
    csv_file_name = "data/training.csv"

if args.lr is not None:
    lr = float(args.lr)
else:
    lr = 0.01

if args.epsilon is not None:
    epsilon = float(args.epsilon)
else:
    epsilon = 0.95

if args.gamma is not None:
    gamma = float(args.gamma)
else:
    gamma = 0.95

if args.enemy_elo is not None:
    enemy_player = initializeStockfish(int(args.enemy_elo))
else:
    enemy_player = initializeStockfish(1000)

# !!! Most important part: Decides if we train or create new examples !!!
if args.mode is not None:
    if args.mode == "train":
        train(n_epochs, batch_size, lr)
    elif args.mode =="examples":
        create_new_examples(10, 20, load_model(model_name))
else:
    print("No mode was chosen, nothing is done in reinforcement.py")

