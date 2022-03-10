'''
Simulation / Maze constants important for the Pacman problem

[!] IMPORTANT:
  - YOU MUST NOT TOUCH THIS FILE AT ALL, NO EDITS OR ADDITIONS!
    Any changes will be overwritten during testing
  - If you need additional constants shared between your files,
    make your own damn module
'''

import torch

class Constants:

    # Simulation constants
    N_SIMS = 8000
    MAX_MOVES = 200
    TICK_LEN = 0 # in ms
    GHOST_EPSILON = 0.1
    DEBUG = False
    VERBOSE = False
    GUI = False
    TRAINING = True
    
    # Training Constants
    BATCH_SIZE = 32
    GAMMA = 0.95
    EPS_GREEDY = 0.1
    TARGET_UPDATE = 100
    MEM_SIZE = 10000
    PARAM_PATH = "./dat/params.pth"
    MEM_PATH = "./dat/mem.pkl"

    # Movement constants + location modifiers
    MOVES = ["U", "D", "L", "R"]
    MOVE_DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

    # Maze content constants
    WALL_BLOCK = "X"
    GHOST_BLOCK = "G"
    PELLET_BLOCK = "O"
    SAFE_BLOCK = "."
    PLR_BLOCK = "P"
    DEATH_BLOCK = "D"
    WIN_BLOCK = "W"
    TIMEOUT_BLOCK = "T"
    ENTITIES = [WALL_BLOCK, PELLET_BLOCK, PLR_BLOCK]
    
    # Used to determine whether GPU acceleration is available or not
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Active Maze Environment
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    