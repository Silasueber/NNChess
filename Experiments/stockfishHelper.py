import platform
from stockfish import Stockfish

def initializeStockfish(elo=None):
    stockfish = None
    if platform.system() == 'Windows':
        stockfish = Stockfish(
        path="C:\\Uni\\Siena_Studium\\Neural Nets\\projects\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")
    else:
        stockfish = Stockfish()

    if elo is not None:
        stockfish.set_elo_rating(elo)
    return stockfish
