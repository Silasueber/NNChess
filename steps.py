# Step 0: Think about the best way to represent the data

# Step 0.a: every piece is represented by a number of its value [Pawn 1, Knight 3, ...]
# Step 0.b: every piece is represented by an one-hot-encoding [Pawn [1,0,0,0,0,0], Knight [0,1,0,0,0,0]]
# Step 0.c: how do we represent which players turn it is? With an extra neuron?

# Step 1: create a Dataset

# Step 1.1: create a Dataset with the labels of 1 [White winning] 0 [Draw] and -1 [Black winning]
# Step 1.1.a: create a Dataset with stockfish playing against each other and evaluating every position
# Step 1.1.b: create a Dataset with positions from lichess dataset or other datasets
# Step 1.1.c: create a Dataset with games from only one player (Magnus Carlsen, Ding Liren, ...)

# Step 1.2: create a Dataset with the real labels of stockfish [-15 -> 15]
# Step 1.2.a: create a Dataset with stockfish playing against each other and evaluating every position
# Step 1.2.b: create a Dataset with positions from lichess dataset or other datasets
# Step 1.2.c: create a Dataset with games from only one player (Magnus Carlsen, Ding Liren, ...)

# Step 2: train a neural network to eval the positions

#Step 3: Reinformcement learning

# Databank of games
# Mate in 1 for black: "8/8/8/8/8/6k1/2q5/6K1 w - - 0 1"

# Games that get played "correctly"
# p3_2_1_night.pt 8/2k5/6K1/8/8/4Q3/3R4/8 w - - 0 1 Mate in 3
# 10000_chess.pt 6r1/5n2/8/2K5/7k/8/4Q3/8 w - - 0 1 Long but nice game (still huge blunders)

# IDEA 
# Create new dataset where there is also a node for checks