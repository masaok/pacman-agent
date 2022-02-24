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

    BATCH_SIZE = 32
    GAMMA = 0.999
    EPS_GREEDY = 0.1
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
        self.steps = 0
        if exists(Constants.DEVICE):
            self.pol_net.load_state_dict(torch.load(Constants.PARAM_PATH))
            self.tar_net.load_state_dict(pol_net.state_dict())
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
        if random.random() < PacmanAgent.EPS_GREEDY:
            return random.choice(Constants.MOVES)
        maze_vectorized = ReplayMemory.vectorize_maze(perception)
        move_probs = list(self.pol_net(maze_vectorized))
        move_probs = {move: move_probs[moveIdx] for moveIdx, move in enumerate(Constants.MOVES)}
        move_probs = {move: prob for (move, prob) in move_probs.items() if move in {s[0] for s in legal_actions}}
        return max(move_probs, key=move_probs.get) if len(move_probs) > 0 else random.choice(legal_actions.keys())
    
    def get_reward(self, state, action, next_state):
        last_maze = MazeProblem(state)
        curr_maze = MazeProblem(next_state)
        reward = -0.01
        
        if len(curr_maze.get_pellets()) < len(last_maze.get_pellets()):
            reward += 1
        
        if not curr_maze.get_win_state() == None:
            reward += 1
            
        if not curr_maze.get_death_state() == None:
            reward -= 10
        
        return reward
    
    def give_transition(self, state, action, next_state):
        reward = torch.tensor([self.get_reward(state, action, next_state)], device=Constants.DEVICE)
        self.memory.push(
            ReplayMemory.vectorize_maze(state), 
            ReplayMemory.vectorize_move(action), 
            ReplayMemory.vectorize_maze(next_state), 
            reward
        )
        self.optimize_model()
        self.steps += 1
        if self.steps % PacmanAgent.TARGET_UPDATE == 0:
            self.tar_net.load_state_dict(self.pol_net.state_dict())
            torch.save(self.pol_net.state_dict(), Constants.PARAM_PATH)
    
    def optimize_model(self):
        if len(self.memory) < PacmanAgent.BATCH_SIZE:
            return
        episodes = self.memory.sample(PacmanAgent.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of episodes
        # to Transition of batch-arrays.
        # batch = Episode(*zip(*episodes))
    
        # TODO: Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.pol_net
        state_action_values = torch.zeros(len(Constants.MOVES), device=Constants.DEVICE)
        for e in episodes:
            action = e.action.tolist()
            action_index = action.index(1)
            state_action_values[action_index] += self.pol_net(e.state)[action_index]
        # state_action_values = [self.pol_net(e.state).mask_select(0, e.action.to(torch.int64)) for e in episodes]
        print("Q(s,a): ", state_action_values)
        next_state_action_values = torch.tensor([self.tar_net(e.next_state).max(0)[0] for e in episodes], device=Constants.DEVICE)
        print("Q(s',a'): ", next_state_action_values)
        rewards = torch.tensor([e.reward for e in episodes], device=Constants.DEVICE)
        print("R: ", rewards)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.tar_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        expected_state_action_values = (next_state_action_values * PacmanAgent.GAMMA) + rewards
        print("Target: ", expected_state_action_values)
    
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.pol_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
