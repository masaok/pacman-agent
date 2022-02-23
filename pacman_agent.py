'''
Pacman Agent employing a PacNet trained in another module to
navigate perilous ghostly pellet mazes.
'''

import time
import random
import numpy as np
import torch
from os.path import exists
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from reinforcement_trainer import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNets trained in the pac_trainer.py
    module.
    '''

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, really just needs the model and
        its plan Queue.
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained.
        """
        
        # TODO: What if params don't exist or aren't from the right device?
        self.memory = ReplayMemory(1000)
        self.pol_net = PacNet(maze).to(Constants.DEVICE)
        self.tar_net = PacNet(maze).to(Constants.DEVICE)
        if exists(Constants.DEVICE):
            self.pol_net.load_state_dict(torch.load(Constants.PARAM_PATH))
            self.tar_net.load_state_dict(pol_net.state_dict())
        self.pol_net.eval()
        self.tar_net.eval()

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: Action choice from the set of legal_actions
        """
        maze_vectorized = ReplayMemory.vectorize_maze(perception)
        move_probs = list(self.pol_net(maze_vectorized))
        move_probs = {move: move_probs[moveIdx] for moveIdx, move in enumerate(Constants.MOVES)}
        move_probs = {move: prob for (move, prob) in move_probs.items() if move in {s[0] for s in legal_actions}}
        return max(move_probs, key=move_probs.get) if len(move_probs) > 0 else random.choice(legal_actions.keys())
    
    def give_transition(self, state, action, next_state):
        return
    
    def optimize_model():
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=Constants.DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.pol_net
        state_action_values = self.pol_net(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.tar_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.tar_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.pol_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
