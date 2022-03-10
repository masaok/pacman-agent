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

Episode = namedtuple('Episode', ('state', 'action', 'next_state', 'reward', 'is_terminal'))

class ReplayMemory(object):
    
    maze_entity_indexes = {entity: index for index, entity in enumerate(Constants.ENTITIES)}
    move_indexes = {move: index for index, move in enumerate(Constants.MOVES)}

    def __init__(self, capacity):
        '''
        Initializes the ReplayMemory object with a deque that will never exceed
        the given capacity. Older memories are replaced by newer ones once the memory
        reaches this size.
        :capacity: Max number of episodes to hold in the memory
        '''
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''
        Adds a new episode to this ReplayMemory
        '''
        self.memory.append(Episode(*args))

    def sample(self, batch_size):
        '''
        Returns a batch of episodes of the requested batch_size
        :batch_size: the number of episodes to return
        :returns: list of batch_size randomly sampled episodes
        '''
        return random.sample(self.memory, batch_size)
    
    def save (self):
        '''
        Saves the current ReplayMemory's episodes to disk so that they may
        persist between training sessions. Saved to Constants.MEM_PATH
        '''
        mem_file = open(Constants.MEM_PATH, "wb")
        pickle.dump(self.memory, mem_file)
        mem_file.close()
        
    def load(self):
        '''
        Loads a ReplayMemory's episodes from disk if it exists at Constants.MEM_PATH
        '''
        mem_file = open(Constants.MEM_PATH, "rb")
        self.memory = pickle.load(mem_file)
        mem_file.close()

    def __len__(self):
        '''
        :returns: the number episodes currently stored in the ReplayMemory 
        '''
        return len(self.memory)
    
    def vectorize_maze(maze):
        '''
        Converts the raw input maze (some Strings representing the Maze
        entities as specified in Constants.ENTITIES) into the vectorized
        input layer for the PacNet.
        :maze: String grid representation of the maze and its entities
        :returns: N x rows x cols numpy matrix where N = the number of maze
        entities specified in Constants.ENTITIES
        '''
        rows = len(maze)
        cols = len(maze[0])
        result = np.zeros((len(Constants.ENTITIES), rows, cols), dtype=np.int8)
        for r, row in enumerate(maze):
            for c, cell in enumerate(row):
                if cell in Constants.ENTITIES:
                    result[ReplayMemory.maze_entity_indexes[cell]][r][c] = 1.0
        
        return result
    
    def vectorize_move(move):
        '''
        Converts the given move from the possibilities of Constants.MOVES to
        the one-hot pytorch tensor representation.
        :move: String representing an action to be taken
        :returns: One-hot vector representation of that action.
        '''
        return F.one_hot(torch.tensor(ReplayMemory.move_indexes[move]), num_classes=len(ReplayMemory.move_indexes)).to(torch.float).to(Constants.DEVICE)
    
    def move_vec_to_index(vec):
        '''
        Converts the given 1-hot encoded vector of a move into its corresponding
        integer index.
        :vec: Vector representation of a move
        :returns: Index of that vector
        '''
        return list(vec).index(1)


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
        conv1_out, conv2_out = 32, 64
        fc_out = 512
        
        # Determine output size of convolutional layer as a function of maze size
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(cols)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(rows)))
        linear_input_size = convw * convh * conv2_out
        
        # Two convolutional layers
        self.conv1 = nn.Conv2d(entities, conv1_out, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=2, stride=1)
        # Followed by a dense layer
        self.fc3 = nn.Linear(linear_input_size, fc_out)
        self.fc4 = nn.Linear(fc_out, moves)
        self.head = nn.Linear(linear_input_size, moves)


    def forward(self, x):
        """
        Computes the output of the PacNet for input maze x
        :x: Raw state input at the first layer of the neural network
        :returns: Output Q(s,a) activations for each a
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x

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
        outcome = Environment.run_game(debug=Constants.DEBUG, verbose=Constants.VERBOSE, step=False, gui=Constants.GUI)
        wins += outcome["win"]
        win_ema = (1-ema_alpha) * win_ema + (ema_alpha) * outcome["win"]
        move_ema = (1-ema_alpha) * move_ema + (ema_alpha) * (outcome["moves"] / Constants.MAX_MOVES)
        pell_ema = (1-ema_alpha) * pell_ema + (ema_alpha) * (outcome["pellets"] / outcome["max_pellets"])
        print("  [M] Moves:\t", outcome["moves"], "\t", move_ema)
        print("  [P] Pellets:\t", outcome["pellets"], "\t", pell_ema)
        print("  [W] Wins:\t", wins, "\t", win_ema)
        plot_wins.append(win_ema)
        plot_moves.append(move_ema)
        plot_pells.append(pell_ema)
    
    plt.figure(1)
    plt.clf()
    plt.title("PacNet RL")
    plt.xlabel("Game")
    plt.plot(plot_wins, label="Win %")
    plt.plot(plot_moves, label="Max Move %")
    plt.plot(plot_pells, label="Pellet %")
    plt.legend(["Win %", "Max Move %", "Pellet %"])
    plt.ioff()
    plt.show()
