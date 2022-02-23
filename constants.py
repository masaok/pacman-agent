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
    MAX_MOVES = 100
    GHOST_EPSILON = 0.1

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
    TIME_BLOCK = "T"
    ENTITIES = [WALL_BLOCK, GHOST_BLOCK, PELLET_BLOCK, SAFE_BLOCK, PLR_BLOCK]
    
    # Used to determine whether GPU acceleration is available or not
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training and Maze Generation constants
    PARAM_PATH = "./dat/params.pth" # Determines parameter save location
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "XG.XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    