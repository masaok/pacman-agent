"""
Introductory Deep Learning exercise for training agents to navigate
small Pacman Mazes
"""

import time
import random
import re
import pandas as pd
import numpy as np
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
                        ('state', 'action', 'next_state', 'reward', 'is_terminal'))

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
    
    def move_vec_to_index(vec):
        return list(vec).index(1)
    
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
        rows = len(maze)
        cols = len(maze[0])
        result = np.zeros((len(Constants.ENTITIES), rows, cols), dtype=np.int8)
        for r, row in enumerate(maze):
            for c, cell in enumerate(row):
                if cell in Constants.ENTITIES:
                    result[ReplayMemory.maze_entity_indexes[cell]][r][c] = 1.0
        
        return result
#         return torch.from_numpy(result).to(torch.float).unsqueeze(0).to(Constants.DEVICE)
    
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
        
        # Convolutional and pooling layers
#         self.conv1 = nn.Conv3d(entities, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm3d(16)
#         self.conv2 = nn.Conv3d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm3d(32)
#         self.conv3 = nn.Conv3d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm3d(32)

        self.conv1 = nn.Conv2d(entities, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        
        # Dense layer with outputs
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(cols)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(rows)))
        linear_input_size = convw * convh * 64
        
        self.fc3 = nn.Linear(2240, 512)
        self.fc4 = nn.Linear(512, moves)
#         self.head = nn.Linear(linear_input_size, moves)


    def forward(self, x):
        """
        Computes the output of the PacNet for input maze x
        :x: Raw input vector at the first layer of the neural network
        :returns: Output Q(s,a) activations for each a
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(torch.flatten(x))

if __name__ == "__main__":
    wins = 0
    win_ema = 0
    move_ema = 0
    pell_ema = 0
    ema_alpha = 0.1
    plot_wins = []
    plot_moves = []
    plot_pells = []
    for i_episode in range(Constants.N_SIMS):
        print("==============================")
        print("Iteration " + str(i_episode))
        print("==============================")
        # Initialize the environment and state
        outcome = Environment.run_game(debug=Constants.DEBUG, verbose=Constants.VERBOSE, step=False, gui=Constants.GUI)
        wins += outcome["win"]
        win_ema = (1-ema_alpha) * win_ema + (ema_alpha) * outcome["win"]
        move_ema = (1-ema_alpha) * move_ema + (ema_alpha) * (outcome["moves"] / Constants.MAX_MOVES)
        pell_ema = (1-ema_alpha) * pell_ema + (ema_alpha) * (outcome["pellets"] / outcome["max_pellets"])
        print("  [M] Moves:\t\t\t", outcome["moves"], move_em)
        print("  [P] Pellets:\t\t\t", outcome["pellets"], pell_ema)
        print("  [W] Wins:\t\t\t", wins, win_ema)
        plot_wins.append(win_ema)
        plot_moves.append(move_ema)
        plot_pells.append(pell_ema)
    
    plt.figure(1)
    plt.clf()
    plt.title("Test")
    plt.xlabel("Episode")
    plt.plot(plot_wins, label="Win %")
    plt.plot(plot_moves, label="Max Move %")
    plt.plot(plot_pells, label="Pellet %")
    plt.legend(["Win %", "Max Move %", "Pellet %"])
    plt.ioff()
    plt.show()
