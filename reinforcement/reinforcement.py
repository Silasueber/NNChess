import chess
import torch.nn as nn
from stockfishHelper import initializeStockfish
import stockfish
# Workflow mainly from here: https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# Some GPT conversation: https://chat.openai.com/share/296512a2-3cf9-4378-8bb9-992b3f3c110c
csv_file_name = "data/training.csv"
evaluator = initializeStockfish()


# Workflow 1.:
possible_pieces = 7 * 2 + 1 #each player has 7 unique pieces (King, Queen, Pawn, Knight, Rook, 2x Bishop (different fields) and empty field
fields_in_chess_board = 8*8
input_size = possible_pieces*fields_in_chess_board
no_of_white_pieces = 14 #we play only white, therefore only need white actions
output_size = no_of_white_pieces * fields_in_chess_board # simplified assumptuion that every piece can move everywhere during the game
q_net = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLu(),
    nn.Linear(32, output_size)
)
target_net = q_net # copy q_net to target net

#2 Workflow --> Generate data

# See how much functionality we need, otherwise put in own class/file
def create_new_example(state):
    # get legal moves
    # select with epsilon greedy
    print(f"save state, action, reward, next state to {csv_file_name}")


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