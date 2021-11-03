'''
The Pathfinder class is responsible for finding a solution (i.e., a
sequence of actions) that takes the agent from the initial state to the
given goal

This task is done in the solve method, as parameterized
by a maze pathfinding problem, and is aided by the SearchTreeNode DS.
'''

import queue
import unittest
import itertools
import math
from maze_problem import *
from search_tree_node import *

def _get_solution(node):
    """
    Returns a solution (a sequence of actions) from the given
    SearchTreeNode node, presumed to be a goal
    :node: A goal SearchTreeNode in the A* Search Tree 
    """
    soln = []
    cum_cost = node.totalCost
    while node.parent is not None:
        soln.append(node.action)
        node = node.parent
    soln.reverse()
    return (cum_cost, soln)

def heuristic(state, goal):
    """
    Implements the Manhattan Distance Heuristic, which (given a state)
    provides the cell-distance to the nearest goal state
    :state: A maze location tuple
    :goal: A maze location tuple
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def pathfind(problem, start, dest):
    """
    Given a MazeProblem, a start location tuple, and a destination tuple,
    returns the sequence of actions that takes the agent from the start
    to the destination via the A* algorithm
    :problem: A MazeProblem object
    :start: A maze location tuple
    :dest: A maze location tuple
    """
    # Setup
    frontier  = queue.PriorityQueue()
    closedSet = set()
    
    # Search!
    frontier.put(SearchTreeNode(start, None, None, 0, heuristic(start, dest)))
    while not frontier.empty():
        # Get front node of priority queue
        expanding = frontier.get()
        
        # Test for goal state
        if expanding.state == dest:
            return _get_solution(expanding)
        
        # Compute evaluation function for expanded node, f(n) = g(n) + h(n)
        evaluation = expanding.evaluation()
        
        # Add expanded node to closedSet
        closedSet.add(expanding.state)
        
        # Generate new nodes on frontier
        for (action, cost, nextState) in problem.transitions(expanding.state):
            childTotalCost = expanding.totalCost + cost
            childHeuristicCost = heuristic(nextState, dest)
            if nextState in closedSet:
                continue
            frontier.put(SearchTreeNode(nextState, action, expanding, childTotalCost, childHeuristicCost))
    
    # No solution
    return None

