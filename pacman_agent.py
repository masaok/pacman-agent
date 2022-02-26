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
from maze_problem import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNets trained in the pac_trainer.py
    module.
    '''

    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_GREEDY = 0.1
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 50
    MEM_SIZE = 1000

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, really just needs the model and
        its plan Queue.
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained.
        """
        
        # TODO: What if params don't exist or aren't from the right device?
        self.memory = ReplayMemory(PacmanAgent.MEM_SIZE)
        self.pol_net = PacNet(maze).to(Constants.DEVICE)
        self.tar_net = PacNet(maze).to(Constants.DEVICE)
        self.steps = 0
        self.prev_state = torch.zeros(self.pol_net.maze_vec_dims, device=Constants.DEVICE)
        if exists(Constants.PARAM_PATH):
            self.pol_net.load_state_dict(torch.load(Constants.PARAM_PATH))
            self.tar_net.load_state_dict(self.pol_net.state_dict())
        if exists(Constants.MEM_PATH):
            self.memory.load()
        # self.pol_net.eval()
        self.tar_net.eval()
        self.optimizer = torch.optim.RMSprop(self.pol_net.parameters())

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: Action choice from the set of legal_actions
        """
        legal_actions = dict(legal_actions)
        if random.random() < PacmanAgent.EPS_GREEDY:
            return random.choice(list(legal_actions.keys()))
        curr_state = ReplayMemory.vectorize_maze(perception)
#         maze_vectorized = torch.cat((self.prev_state, curr_state), 0)
        maze_vectorized = curr_state
        move_probs = list(self.pol_net(maze_vectorized))
        move_probs = {move: move_probs[moveIdx] for moveIdx, move in enumerate(Constants.MOVES)}
        move_probs = {move: prob for (move, prob) in move_probs.items() if move in {s[0] for s in legal_actions}}
        return max(move_probs, key=move_probs.get) if len(move_probs) > 0 else random.choice(list(legal_actions.keys()))
    
    def get_reward(self, state, action, next_state):
        last_maze = MazeProblem(state)
        curr_maze = MazeProblem(next_state)
        reward = -0.1
        
        if len(curr_maze.get_pellets()) < len(last_maze.get_pellets()):
            reward += 1
        
        if not curr_maze.get_win_state() is None:
            reward += 10
            
        if not curr_maze.get_death_state() is None:
            reward -= 10
        
        return reward
    
    def give_transition(self, state, action, next_state, is_terminal):
        reward = torch.tensor([self.get_reward(state, action, next_state)], device=Constants.DEVICE)
        state_vec = ReplayMemory.vectorize_maze(state)
        self.memory.push(
            self.prev_state,
            state_vec,
            ReplayMemory.vectorize_move(action), 
            ReplayMemory.vectorize_maze(next_state), 
            reward,
            is_terminal
        )
        self.optimize_model()
        self.steps += 1
        if self.steps % PacmanAgent.TARGET_UPDATE == 0:
            self.tar_net.load_state_dict(self.pol_net.state_dict())
        self.prev_state = state_vec
    
    def give_terminal(self):
        torch.save(self.pol_net.state_dict(), Constants.PARAM_PATH)
        self.memory.save()
    
    def optimize_model(self):
        if len(self.memory) < PacmanAgent.BATCH_SIZE:
            return
        episodes = self.memory.sample(PacmanAgent.BATCH_SIZE)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.pol_net
        state_action_values = torch.zeros(len(Constants.MOVES), device=Constants.DEVICE)
        target_action_values = torch.zeros(len(Constants.MOVES), device=Constants.DEVICE)
        action_indexes = torch.tensor([e.action.tolist().index(1) for e in episodes])
        for e in episodes:
            action = e.action.tolist()
            action_index = action.index(1)
#             state_action_value = self.pol_net(torch.cat((e.prev_state, e.state), 0))[action_index]
            state_action_value = self.pol_net(e.state)[action_index]
            next_state_action_value = 0 if e.is_terminal else self.tar_net(e.next_state).max(0)[0]
#             next_state_action_value = 0 if e.is_terminal else self.tar_net(torch.cat((e.state, e.next_state), 0)).max(0)[0]
#             next_state_action_value = self.tar_net(torch.cat((e.state, e.next_state), 0)).max(0)[0]
#             next_state_action_value = self.tar_net(torch.cat((e.state, e.next_state), 0)).max(0)[0]
            target_action_value = (next_state_action_value * PacmanAgent.GAMMA) + e.reward
            state_action_values[action_index] += state_action_value
            target_action_values[action_index] += target_action_value[0]
            
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
#         print("Q(s,a)= ", state_action_values)
#         print("Target= ", target_action_values)
        loss = criterion(state_action_values, target_action_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.pol_net.parameters():
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
