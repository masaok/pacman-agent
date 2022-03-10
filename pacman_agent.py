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
    Deep learning Pacman agent that employs PacNet DQNs.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, this includes initializing the
        policy DQN (+ target DQN, ReplayMemory, and optimizer if training) and
        any other 
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained.
        """
        self.memory = ReplayMemory(Constants.MEM_SIZE)
        self.pol_net = PacNet(maze).to(Constants.DEVICE)
        self.tar_net = PacNet(maze).to(Constants.DEVICE)
        self.steps = 0
        if exists(Constants.PARAM_PATH):
            self.pol_net.load_state_dict(torch.load(Constants.PARAM_PATH))
            self.tar_net.load_state_dict(self.pol_net.state_dict())
        if exists(Constants.MEM_PATH):
            self.memory.load()
        self.tar_net.eval()
        self.optimizer = torch.optim.Adam(self.pol_net.parameters())

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available. If training,
        must manage the explore vs. exploit dilemma through some form of ASR.
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: Action choice from the set of legal_actions
        """
        legal_actions = dict(legal_actions)
        if random.random() < Constants.EPS_GREEDY and Constants.TRAINING:
            return random.choice(list(legal_actions.keys()))
        curr_state = ReplayMemory.vectorize_maze(perception)
        maze_vectorized = torch.from_numpy(np.stack(curr_state)).unsqueeze(0).to(torch.float).to(Constants.DEVICE)
        move_probs = self.pol_net(maze_vectorized)[0].tolist()
        move_probs = {move: move_probs[moveIdx] for moveIdx, move in enumerate(Constants.MOVES)}
        move_probs = {move: prob for (move, prob) in move_probs.items() if move in {s[0] for s in legal_actions}}
        return max(move_probs, key=move_probs.get) if len(move_probs) > 0 else random.choice(list(legal_actions.keys()))
    
    def get_reward(self, state, action, next_state):
        '''
        The reward function that determines the numerical desirability of the
        given transition from state -> next_state with the chosen action.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :returns: R(s, a, s') for the given transition
        '''
        last_maze = MazeProblem(state)
        curr_maze = MazeProblem(next_state)
        reward = -0.1
        
        if len(curr_maze.get_pellets()) < len(last_maze.get_pellets()):
            reward += 10
        
        if not curr_maze.get_win_state() is None:
            reward += 100
            
        if not curr_maze.get_death_state() is None:
            reward -= 100
        
        return reward
    
    def give_transition(self, state, action, next_state, is_terminal):
        '''
        Called by the Environment after both Pacman and ghosts have moved on a
        given turn, supplying the transition that was observed, which can then
        be added to the training agent's memory and the model optimized. Also
        responsible for periodically updating the target network.
        [!] If not training, this method should do nothing.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :is_terminal: whether or not next_state is a terminal state
        '''
        if not Constants.TRAINING:
            return
        reward_val = self.get_reward(state, action, next_state)
        reward = torch.tensor([reward_val], device=Constants.DEVICE)
        mem_weight = 1 if reward_val < 0 else 10
        for m in range(mem_weight):
            self.memory.push(
                ReplayMemory.vectorize_maze(state),
                ReplayMemory.vectorize_move(action),
                ReplayMemory.vectorize_maze(next_state),
                reward,
                is_terminal
            )
        self.optimize_model()
        self.steps += 1
        if self.steps % Constants.TARGET_UPDATE == 0:
            self.tar_net.load_state_dict(self.pol_net.state_dict())
    
    def give_terminal(self):
        '''
        Called by the Environment upon reaching any of the terminal states:
          - Winning (eating all of the pellets)
          - Dying (getting eaten by a ghost)
          - Timing out (taking more than Constants.MAX_MOVES number of turns)
        Useful for cleaning up fields, saving weights and memories to disk if
        desired, etc.
        [!] If not training, this method should do nothing.
        '''
        if not Constants.TRAINING:
            return
        torch.save(self.pol_net.state_dict(), Constants.PARAM_PATH)
        self.memory.save()
    
    def optimize_model(self):
        '''
        Primary workhorse for training the policy DQN. Samples a mini-batch of
        episodes from the ReplayMemory and then takes a step of the optimizer
        to train the DQN weights.
        [!] If not training OR fewer episodes than Constants.BATCH_SIZE have
        been recorded, this method should do nothing.
        '''
        if len(self.memory) < Constants.BATCH_SIZE:
            return
        
        # Set up batch variables
        episodes = self.memory.sample(Constants.BATCH_SIZE)
        batch_s, batch_a, batch_n, batch_r, batch_t = zip(*episodes)
        batch_s = torch.from_numpy(np.stack(batch_s)).to(torch.float).to(Constants.DEVICE)
        batch_r = torch.DoubleTensor(batch_r).unsqueeze(1).to(Constants.DEVICE)
        batch_a = torch.LongTensor(list(map(ReplayMemory.move_vec_to_index, batch_a))).unsqueeze(1).to(Constants.DEVICE)
        batch_n = torch.from_numpy(np.stack(batch_n)).to(torch.float).to(Constants.DEVICE)
        batch_t = torch.ByteTensor(batch_t).unsqueeze(1).to(Constants.DEVICE)
    
        # Q(s, a) for each episode's state s and chosen action a
        state_action_values = self.pol_net(batch_s).gather(1, batch_a)
        
        # Finding max_a' Q(s', a') for each episode based on the target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s == False, batch_t)), device=Constants.DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for i,s in enumerate(batch_n) if not batch_t[i]])
        next_state_action_values = torch.zeros(Constants.BATCH_SIZE, device=Constants.DEVICE)
        next_state_action_values[non_final_mask] = self.tar_net(non_final_next_states).detach().max(1)[0]
        next_state_action_values = next_state_action_values.unsqueeze(1)
        
        # Completing the temporal difference update: R(s, a, s') + GAMMA * max_a' Q(s', a')
        target_action_values = (next_state_action_values * Constants.GAMMA) + batch_r
        
        # Compute Huber loss between actual and expected Q-values
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target_action_values)
    
        # Optimize the model based on the computed loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
