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
    def choose_action(self, perception):
        """
        Returns an action from a set {U, D, L, R} given perception (maze)
        Currently returns a random choice
        TODO: give a choice of moving toward the goal or away from the goal

        Parameters:
            perception (array): array of strings representation of the maze

        Returns:
            str: direction to move
        """

        directions = ["U", "D", "L", "R"]
        random_index = random.randint(0,len(directions)-1)
        return directions[random_index]


    