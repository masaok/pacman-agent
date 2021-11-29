'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

import time
import random
import numpy as np
import torch
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from pac_trainer import *
# [!] TODO: import your trained model when ready here!


class PacmanAgent:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze):
        # The agent's plan will be a queue storing the sequence of
        # actions that the environment will execute
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PacNet(maze).to(device)
        self.model.load_state_dict(torch.load(Constants.PARAM_PATH))
        self.model.eval()
        self.plan = Queue()

        # [!] TODO: Initialize any other knowledge-related attributes for
        # agent here, or any other record-keeping attributes you'd like

    ##################################################################
    # Methods
    ##################################################################

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from a set {U, D, L, R} given perception (maze)
        Currently returns a random choice
        TODO: give a choice of moving toward the goal or away from the goal

        Parameters:
            perception (array): array of strings representation of the maze

        Returns:
            str: direction to move
        """
        maze_vectorized = PacmanMazeDataset.vectorize_maze(perception)
        move_probs = list(self.model.forward(maze_vectorized))
        move_probs = {move: move_probs[moveIdx] for moveIdx, move in enumerate(Constants.MOVES)}
        move_probs = {move: prob for (move, prob) in move_probs.items() if move in {s[0] for s in legal_actions}}
        return max(move_probs, key=move_probs.get) if len(move_probs) > 0 else random.choice(legal_actions.keys())
