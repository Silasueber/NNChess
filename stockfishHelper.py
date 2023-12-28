import platform
from stockfish import Stockfish

def initalizeStockfish():
    if platform.system() == 'Windows':
        return Stockfish(
        path="C:\\Uni\\Siena_Studium\\Neural Nets\\projects\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe")
    else:
        return Stockfish()
