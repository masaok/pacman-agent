'''
Specifies a MazeProblem as parameterized by a given grid maze,
assuming that an agent's legal actions can move them one tile in
any cardinal direction
'''

from constants import Constants
import copy

class MazeProblem:
    
    ##################################################################
    # Class Constants
    ##################################################################
    
    # Static COST_MAP for maze components and the cost to move onto them
    # Any maze block not listed here is assumed to have a cost of 1
    # HINT: You can add block types to this!
    COST_MAP = {Constants.GHOST_BLOCK: Constants.get_ghost_penalty()}
    
    def __init__(self, maze):
        """
        Constructs a new pathfinding problem from a maze
        :maze: a list of list of strings containing maze elements
        """
        self.maze = maze
         # Maze block sets
        self._ghosts = set()
        self._pellets = set()
        self._walls = set()

        # Scan for blocks in the input maze
        for (row_num, row) in enumerate(maze):
            for (col_num, cell) in enumerate(row):
                if cell == Constants.WALL_BLOCK:
                    self._walls.add((col_num, row_num))
                if cell == Constants.GHOST_BLOCK:
                    self._ghosts.add((col_num, row_num))
                if cell == Constants.PELLET_BLOCK:
                    self._pellets.add((col_num, row_num))
                if cell == Constants.PLR_BLOCK:
                    self._player_loc = self._initial_loc = (col_num, row_num)

    def get_player_loc(self):
        """
        Returns the current location of Pacman
        """
        return copy.deepcopy(self._player_loc)
    
    def get_ghosts(self):
        """
        Returns a set containing the locations of all ghosts in the maze
        """
        return copy.deepcopy(self._ghosts)
    
    def get_pellets(self):
        """
        Returns a set containing the locations of all remaining pellets in the maze
        """
        return copy.deepcopy(self._pellets)
    
    def get_walls(self):
        """
        Returns a set containing the locations of all walls in the maze
        """
        return copy.deepcopy(self._walls)
        
    def legal_actions(self, state):
        """
        Returns all legal actions available to a maze agent from the given state
        :state: The current position from which to obtain the legal actions
        :returns: List of types of the format (action, next_state)
        """
        s = state
        possible = [("U", (s[0], s[1]-1)), ("D", (s[0], s[1]+1)), ("L", (s[0]-1, s[1])), ("R", (s[0]+1, s[1]))]
        return [(m[0], m[1]) for m in possible if self.maze[m[1][1]][m[1][0]] != Constants.WALL_BLOCK]
    
    def transitions(self, state):
        """
        Given some state s, the transitions will be represented as a list of tuples
        of the format:
        [(action1, cost_of_action1, result(action1, s)), ...]
        For example, if an agent is at state (1, 1), and can only move right and down
        into clear tiles (.), then the transitions for that s = (1, 1) would be:
        [("R", 1, (2, 1)), ("D", 1, (1, 2))]
        :state: A maze location tuple
        :returns: List of tuples of the format (action, cost, next_state)
        """
        s = state
        possible = self.legal_actions(s)
        return [(s[0], self.cost(s[1]), s[1]) for s in possible]
    
    def cost(self, state):
        """
        Returns the cost of moving onto the given state, and employs
        the MazeProblem's COST_MAP
        :state: A maze location tuple
        :returns:
        """
        cm = MazeProblem.COST_MAP
        cell = self.maze[state[1]][state[0]]
        return cm[cell] if cell in cm else Constants.get_mov_penalty()
    
