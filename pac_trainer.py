'''
Deep Pacman Trainer
'''

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
    
    def __init__(self, training_data):
        self.training_data = training_data
        self.maze_entity_indexes = {entity: index for index, entity in enumerate(Constants.ENTITIES)}
        self.move_indexes = {move: index for index, move in enumerate(Constants.MOVES)}
        print(self.__getitem__(0))

    def __len__(self):
        return len(self.training_data)
    
    def _vectorize_maze(self, maze):
        result = []
        for row in maze:
            for cell in row:
                result.append(self.maze_entity_indexes[cell])
        
        return torch.flatten(F.one_hot(torch.tensor(result, dtype=torch.long), num_classes=len(self.maze_entity_indexes))).to(torch.float)
    
    def _vectorize_move(self, move):
        return F.one_hot(torch.tensor(self.move_indexes[move]), num_classes=len(self.move_indexes)).to(torch.float)

    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        maze, move = row["X"], row["y"]
        return self._vectorize_maze(maze), self._vectorize_move(move)

class PacNet(nn.Module):
    
    def __init__(self, maze):
        super(PacNet, self).__init__()
        self.flatten = nn.Flatten()
        rows = len(maze)
        cols = len(maze[0])
        entities = len(Constants.ENTITIES)
        moves = len(Constants.MOVES)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(rows * cols * entities, 20),
            nn.ReLU(),
            nn.Linear(20, moves),
#             nn.Softmax(),
#             nn.Linear(moves, moves)
        )

    def forward(self, x):
#         x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
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
    maze = ["XXXXXXXXX",
            "X..P....X",
            "X.G..G..X",
            "X..XXXP.X",
            "XP.....PX",
            "X...@...X",
            "XXXXXXXXX"]
    
    result = MazeGen.get_labeled_data(maze, 3000)
    data = PacmanMazeDataset(result)
    train_dataloader = DataLoader(data, batch_size=4, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    
    # NN Construction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PacNet(maze).to(device)
    
    
    # Optimization
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
    print("Done!")
    
#     torch.save(model.state_dict(), "./")
    
