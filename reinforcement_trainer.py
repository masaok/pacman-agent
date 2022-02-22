"""
Introductory Deep Learning exercise for training agents to navigate
small Pacman Mazes
"""

import time
import random
import re
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import namedtuple, deque
from constants import *
from maze_gen import MazeGen

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def vectorize_maze(maze):
        '''
        Converts the raw input maze (some Strings representing the Maze
        entities as specified in Constants.ENTITIES) into the vectorized
        input layer for the PacNet.
        [!] Indicies of maze entities should always correspond to their
            order in Constants.ENTITIES; see maze_entity_indexes map as
            a convenient tool for ensuring this.
        [!] Used in both training and deployment
        :maze: String grid representation of the maze and its entities
        :returns: 1-D numerical pytorch tensor representing the maze
        '''
        result = []
        for row in maze:
            for cell in row:
                result.append(PacmanMazeDataset.maze_entity_indexes[cell])
        
        return torch.flatten(F.one_hot(torch.tensor(result, dtype=torch.long), num_classes=len(PacmanMazeDataset.maze_entity_indexes))).to(torch.float).to(Constants.DEVICE)
    
    def vectorize_move(move):
        '''
        Converts the given move from the possibilities of Constants.MOVES to
        the one-hot pytorch tensor representation.
        [!] Indicies of moves should always correspond to their
            order in Constants.MOVES; see move_indexes map as a convenient
            tool for ensuring this.
        [!] Used in both training and deployment
        :move: String representing an action to be taken
        :returns: One-hot vector representation of that action.
        '''
        return F.one_hot(torch.tensor(PacmanMazeDataset.move_indexes[move]), num_classes=len(PacmanMazeDataset.move_indexes)).to(torch.float).to(Constants.DEVICE)


class PacNet(nn.Module):
    """
    PyTorch Neural Network extension for the Pacman gridworld, which is fit to a
    particular maze configuration (walls, and optionally pellets, are in fixed spots)
    See: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    """
    
    def __init__(self, maze):
        """
        Initializes a PacNet for the given maze, which has maze-specific configuration
        requirements like the number of rows and cols. Used to perform imitation learning
        to select one of the 4 available actions in Constants.MOVES in response to the
        positional maze entities in Constants.ENTITIES
        :maze: The Pacman Maze structure on which this PacNet will be trained
        """
        super(PacNet, self).__init__()
        self.flatten = nn.Flatten()
        rows = len(maze)
        cols = len(maze[0])
        entities = len(Constants.ENTITIES)
        moves = len(Constants.MOVES)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(rows * cols * entities, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, moves),
        )

    def forward(self, x):
        """
        Computes the output of the PacNet for input maze x
        :x: Raw input vector at the first layer of the neural network
        :returns: Output activations
        """
        logits = self.linear_relu_stack(x)
        return logits


    
