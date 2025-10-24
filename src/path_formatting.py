import pathlib
from pathlib import Path

def w2l(path):
    # Remove the drive letter "X:" and replace backslashes
    # w2l(r'X:\eva\data\processed\eb02\eb02_20250717\kilosort')
    path = path.replace("X:\\", "/storage3/").replace("\\", "/")
    return path