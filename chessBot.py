import chess 
import torch
import time


amount_to_model = 0

position_eval = {}
board = chess.Board("8/8/3kn3/8/8/4QK2/8/8 w - - 0 1") 
model = torch.load("models/p3_2_1_night.pt")

piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 1000}

def convertPositionToString(board):
    piece_values = {'p': 1, 'r': 5, 'n': 3, 'b': 3, 'q': 10, 'k': 1000}
    board = str(board)
    lines = board.split('\n')

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

def evalPosition():
    global amount_to_model
    whites_turn = board.turn
    position = ("1," + convertPositionToString(board)if whites_turn else "0," + convertPositionToString(board))
    if board.fen() in position_eval:
        return position_eval[board.fen()]
    test = position.split(",")
    test = [int(t) for t in test]
    test = torch.tensor(test, dtype=torch.float32)
    predictions = model(test)
    amount_to_model += 1
    # position_eval[board.fen()] = predictions[0].item() - predictions[1].item()
    position_eval[board.fen()] = predictions[0].item() - 0.5
    return  position_eval[board.fen()]

def get_piece_value(piece_type):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return piece_values.get(piece_type, 0)

def order_moves(moves):
    if type(moves) is chess.LegalMoveGenerator:
        move_scores = [0] * moves.count()
    else:
        move_scores = [0] * len(moves)

    for i, move in enumerate(moves):
        move_score_guess = 0
        move_piece_type = board.piece_type_at(move.from_square)
        capture_piece_type = board.piece_type_at(move.to_square)

        if capture_piece_type is not None and capture_piece_type != chess.PAWN:
            move_score_guess = -10 * get_piece_value(capture_piece_type) + get_piece_value(move_piece_type)

        if move.promotion is not None:
            move_score_guess -= get_piece_value(move.promotion)

        move_scores[i] = move_score_guess

    # Sort moves based on their scores
    sorted_moves = [move for _, move in sorted(zip(move_scores, moves), key=lambda x: x[0], reverse=False)]
    return sorted_moves

def SearchCaptures(alpha, beta):
    eval = evalPosition() # EVAL position with ML model here
    if eval >= beta: 
        return beta
    alpha = max(alpha, eval)

    moves = [move for move in board.legal_moves if board.is_capture(move)]
    moves = order_moves(moves)
    for move in moves:
        board.push(move)
        eval = -SearchCaptures(-beta, -alpha)
        board.pop()
        if eval >= beta:
            return beta
        alpha = max(alpha, eval)
    return alpha
    
def minimax(depth, alpha, beta):
    if depth == 0:
        return SearchCaptures(alpha, beta)
    
    moves = board.legal_moves
    moves = order_moves(moves)
    # Either Check or Stalemate
    if len(moves) == 0:
        if board.is_checkmate():
            return -1 -(0.1*depth)
        return 0.5
    value = -2
    for move in moves:
        board.push(move)
        eval = -minimax(depth-1, -beta, -alpha)
        board.pop()
        value = max(eval, value)
        alpha= max(alpha, eval)
        if alpha >= beta: 
            return value
    return value

def getBestMove(b):
    global board
    board = b
    bestValue = -2
    bestMove = None
    moves = board.legal_moves
    moves = order_moves(moves)
    for move in moves:
        board.push(move)
        value = -minimax(3, -2, 2)
        print(str(move) + ":" + str(value))
        board.pop()
        if value >= bestValue:
            bestValue = value
            bestMove = move
    print(bestMove)
    return bestMove

start = time.time()
getBestMove(board)
print(time.time() - start)
print(amount_to_model)
#TODO CANT Checkmate, IDK why