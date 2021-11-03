'''
Simulation / Maze constants important for the BlindBot problem

[!] IMPORTANT:
  - YOU MUST NOT TOUCH THIS FILE AT ALL, NO EDITS OR ADDITIONS!
    Any changes will be overwritten during testing
  - If you need additional constants shared between your files,
    make your own damn module
'''
    
class Constants:
    
    # The following are staticmethods to prevent tampering,
    # I've got my eye on you, even if through this comment
    @staticmethod
    def get_min_score ():
        """
        Returns the minimum score that, if reached, will end the game,
        and bring great shame to your agent
        """
        return -100
    
    @staticmethod
    def get_pit_penalty ():
        """
        Returns the cost of stepping into a Pit... you're not dead just...
        like... really inconvenienced
        """
        return 20
    
    @staticmethod
    def get_pellet_reward ():
        """
        Returns the cost of stepping into a Pit... you're not dead just...
        like... really inconvenienced
        """
        return 1
    
    @staticmethod
    def get_mov_penalty ():
        """
        Returns the cost of a movement onto any safe tile
        """
        return 1

    # Movement constants + location modifiers
    MOVES = ["U", "D", "L", "R"]
    MOVE_DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
    
    # Maze content constants
    WALL_BLOCK  = "X"
    GOAL_BLOCK  = "G"
    PELLET_BLOCK   = "P"
    SAFE_BLOCK  = "."
    PLR_BLOCK   = "@"
    WRN_BLOCK_1 = "1"
    WRN_BLOCK_2 = "2"
    UNK_BLOCK   = "?"
