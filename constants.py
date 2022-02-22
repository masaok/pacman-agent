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

    @staticmethod
    def get_min_score():
        """
        Returns the minimum score that, if reached, will end the game,
        and bring great shame to your agent
        """
        return -100

    @staticmethod
    def get_pellet_reward():
        """
        Returns the reward of stepping on a pellet
        """
        return 0.5

    @staticmethod
    def get_mov_penalty():
        """
        Returns the cost of a movement onto any safe tile
        """
        return 1
    
    @staticmethod
    def get_ghost_penalty():
        """
        Returns the cost of a movement onto a ghost
        """
        return 20
    
    @staticmethod
    def get_ghost_epsilon():
        """
        Returns the likelihood that a Ghost makes a random choice, rather
        than chasing Pacman
        """
        return 0.1

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
    ENTITIES = [WALL_BLOCK, GHOST_BLOCK, PELLET_BLOCK, SAFE_BLOCK, PLR_BLOCK]
    
    # Used to determine whether GPU acceleration is available or not
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training and Maze Generation constants
    PARAM_PATH = "./dat/params.pth" # Determines parameter save location
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    