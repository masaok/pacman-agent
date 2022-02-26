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
import pickle
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import namedtuple, deque
from constants import *
from environment import *
from statistics import *

Episode = namedtuple('Episode',
                        ('prev_state', 'state', 'action', 'next_state', 'reward', 'is_terminal'))

class ReplayMemory(object):
    
    # [!] Class maps that may be useful for vectorizing the maze
    maze_entity_indexes = {entity: index for index, entity in enumerate(Constants.ENTITIES)}
    move_indexes = {move: index for index, move in enumerate(Constants.MOVES)}

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Episode(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def save (self):
        mem_file = open(Constants.MEM_PATH, "wb")
        pickle.dump(self.memory, mem_file)
        mem_file.close()
        
    def load(self):
        mem_file = open(Constants.MEM_PATH, "rb")
        self.memory = pickle.load(mem_file)
        mem_file.close()

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
                if not cell in Constants.ENTITIES:
                    cell = "." # Hack for the environment's mechanics
                result.append(ReplayMemory.maze_entity_indexes[cell])
        
        return torch.flatten(F.one_hot(torch.tensor(result, dtype=torch.long), num_classes=len(ReplayMemory.maze_entity_indexes))).to(torch.float).to(Constants.DEVICE)
    
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
        return F.one_hot(torch.tensor(ReplayMemory.move_indexes[move]), num_classes=len(ReplayMemory.move_indexes)).to(torch.float).to(Constants.DEVICE)


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
        self.maze_vec_dims = rows * cols * entities
        self.linear_relu_stack = nn.Sequential(
#             nn.Linear(self.maze_vec_dims * 2, 256),
            nn.Linear(self.maze_vec_dims, self.maze_vec_dims),
            nn.ReLU(),
            nn.Linear(self.maze_vec_dims, self.maze_vec_dims),
            nn.ReLU(),
            nn.Linear(self.maze_vec_dims, moves),
        )

    def forward(self, x):
        """
        Computes the output of the PacNet for input maze x
        :x: Raw input vector at the first layer of the neural network
        :returns: Output Q(s,a) activations for each a
        """
        q_vals = self.linear_relu_stack(x)
        return q_vals

if __name__ == "__main__":
    win_ema = 0
    move_ema = 0
    ema_alpha = 0.1
    plot_wins = []
    plot_moves = []
    for i_episode in range(Constants.N_SIMS):
        print("==============================")
        print("Iteration " + str(i_episode))
        print("==============================")
        # Initialize the environment and state
        outcome = Environment.run_game(debug=False, step=False, gui=False)
        print("  [M] Moves: ", outcome["moves"])
        win_ema = (1-ema_alpha) * win_ema + (ema_alpha) * outcome["win"]
        move_ema = (1-ema_alpha) * move_ema + (ema_alpha) * (outcome["moves"] / Constants.MAX_MOVES)
        print("  > Win Average: ", win_ema)
        print("  > Moves Average: ", move_ema)
        plot_wins.append(win_ema)
        plot_moves.append(move_ema)
    
    plt.figure(1)
    plt.clf()
    plt.title("Test")
    plt.xlabel("Episode")
    plt.plot(plot_wins)
    plt.plot(plot_moves)
    plt.ioff()
    plt.show()
