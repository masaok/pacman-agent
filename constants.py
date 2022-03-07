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

    # Game-specific constants
    MAX_MOVES = 200
    TICK_LEN = 0 # in ms
    N_SIMS = 2000
    GHOST_EPSILON = 0.1
    DEBUG = False
    VERBOSE = False
    GUI = False
    TRAINING = True

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
    # ENTITIES = [WALL_BLOCK, GHOST_BLOCK, PELLET_BLOCK, SAFE_BLOCK, PLR_BLOCK]
    ENTITIES = [WALL_BLOCK, PELLET_BLOCK, PLR_BLOCK]
    
    # Used to determine whether GPU acceleration is available or not
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training and Maze Generation constants
    PARAM_PATH = "./dat/params.pth" # Determines parameter save location
    MEM_PATH = "./dat/mem.pkl"
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    