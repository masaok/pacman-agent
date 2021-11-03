'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

import time
import random
from pathfinder import *
from maze_problem import *
from queue import Queue

# [!] TODO: import your Problem 1 when ready here!

class PacmanAgent:
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__ (self):
        # self.env  = env
        # self.loc  = env.get_player_loc()
        # self.goal = env.get_goal_loc()
        
        # The agent's maze can be manipulated as a tracking mechanic
        # for what it has learned; changes to this maze will be drawn
        # by the environment and is simply for visuals
        # self.maze = env.get_agent_maze()
        
        # The agent's plan will be a queue storing the sequence of
        # actions that the environment will execute
        self.plan = Queue()
        
        # [!] TODO: Initialize any other knowledge-related attributes for
        # agent here, or any other record-keeping attributes you'd like
    
    
    ##################################################################
    # Methods
    ##################################################################
    
    # [!] TODO! Agent currently just runs straight up
    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.
        
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """
        
        # Agent simply moves randomly at the moment...
        # Do something that thinks about the perception!
        self.plan.put(random.choice(Constants.MOVES))
    
    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()
    