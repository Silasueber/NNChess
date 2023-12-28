import chess 
import torch
from stockfish import Stockfish
# def SearchCaptures(int alpha, int beta):
#     int eval = Evaluate();
#     if (eval >= beta) return beta;
#     alpha = Math.Max(alpha, eval);

#     Move[] moves = OrderMoves(board, board.GetLegalMoves(true));
#     foreach(Move move in moves)
#     {
#         board.MakeMove(move);
#         eval = -SearchCaptures(-beta, -alpha);
#         board.UndoMove(move);
#         if (eval >= beta) return beta;
#         alpha = Math.Max(alpha, eval);
#     }
#     return alpha;

# public int SearchCaptures(int alpha, int beta)
#     {
#         int eval = Evaluate();
#         if (eval >= beta) return beta;
#         alpha = Math.Max(alpha, eval);

#         Move[] moves = OrderMoves(board, board.GetLegalMoves(true));
#         foreach(Move move in moves)
#         {
#             board.MakeMove(move);
#             eval = -SearchCaptures(-beta, -alpha);
#             board.UndoMove(move);
#             if (eval >= beta) return beta;
#             alpha = Math.Max(alpha, eval);
#         }
#         return alpha;
#     }

model = torch.load("models/p2.pt")
board = chess.Board("rnb1kbnr/ppp1pppp/8/3p1q2/8/4N3/PPPPPPPP/RNBQKB1R w KQkq - 0 1") 


def convertPositionToString(fen):
    stock = Stockfish()
    stock.set_fen_position(fen)
    board = stock.get_board_visual()

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

def evalPosition():
    whites_turn = board.turn
    test = ("1," + convertPositionToString(board.fen())if whites_turn else "0," + convertPositionToString(board.fen())).split(",")
    test = [int(t) for t in test]
    test = torch.tensor(test, dtype=torch.float32)
    predictions = model(test)
    return predictions[0].item() - predictions[1].item()



def SearchCaptures(alpha, beta):
    eval = evalPosition() # EVAL position with ML model here
    print(eval)
    if eval >= beta: 
        return beta
    alpha = max(alpha, eval)

    moves = [move for move in board.legal_moves if board.is_capture(move)]
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
    # Either Check or Stalemate
    if moves.count == 0:
        return 0
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

def getBestMove():
    bestValue = -2
    bestMove = None
    moves = board.legal_moves
    for move in moves:
        board.push(move)
        value = -minimax(1, -2, 2)
        print(str(move) + ":" + str(value))
        board.pop()
        if value >= bestValue:
            bestValue = value
            bestMove = move
    return bestMove


    # // Helper function to evaluate all capture moves
    # private Move GetBestMove()
    # {
    #     float bestValue = float.MinValue;
    #     Move bestMove = Move.NullMove;

    #     Move[] moves = OrderMoves(board,board.GetLegalMoves());
    #     foreach (Move move in moves)
    #     {
    #         board.MakeMove(move);
    #         int value = 0;
    #         if (board.IsRepeatedPosition()) value = -100000000;
    #         else value = -minimax(4, int.MinValue, int.MaxValue);

    #         board.UndoMove(move);
    #         if(value >= bestValue)
    #         {
    #             bestValue = value;
    #             bestMove = move;
    #         }
    #     }
    #     return bestMove;

    # }


print(getBestMove())