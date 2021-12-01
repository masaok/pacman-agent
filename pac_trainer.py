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
from constants import *
from maze_gen import MazeGen

class PacmanMazeDataset(Dataset):
    """
    PyTorch Dataset extension used to vectorize Pacman mazes consisting of the
    entities listed in Constants.py to be used for neural network training
    See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    
    # [!] Class maps that may be useful for vectorizing the maze
    maze_entity_indexes = {entity: index for index, entity in enumerate(Constants.ENTITIES)}
    move_indexes = {move: index for index, move in enumerate(Constants.MOVES)}
    
    def __init__(self, training_data):
        self.training_data = training_data
        print(self.__getitem__(0))

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        maze, move = row["X"], row["y"]
        return PacmanMazeDataset.vectorize_maze(maze), PacmanMazeDataset.vectorize_move(move)
    
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
        
        return torch.flatten(F.one_hot(torch.tensor(result, dtype=torch.long), num_classes=len(PacmanMazeDataset.maze_entity_indexes))).to(torch.float).to("cuda")
    
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
        return F.one_hot(torch.tensor(PacmanMazeDataset.move_indexes[move]), num_classes=len(PacmanMazeDataset.move_indexes)).to(torch.float).to("cuda")


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


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    PyTorch Neural Network optimization loop; need not be modified unless tweaks are
    desired.
    See: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    """
    Main method used to load training data, construct PacNet, and then
    train it, finally saving the network's parameters for use by the
    pacman agent.
    See: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    """
    maze = ["XXXXXXXXX",
            "X..O....X",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X...P...X",
            "XXXXXXXXX"]
    
    result = MazeGen.get_labeled_data(maze, Constants.N_SAMPLES)
    data = PacmanMazeDataset(result)
    train_dataloader = DataLoader(data, batch_size=4, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    
    # NN Construction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PacNet(maze).to(device)
    
    # Optimization
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    
    torch.save(model.state_dict(), Constants.PARAM_PATH)
    
