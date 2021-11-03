'''
Environment class responsible for configuring and running
the MazePitfall problem with BlindBot agent
'''

import os
import re
import sys
import time
import copy
from constants import Constants

from pacman_agent import PacmanAgent

class Environment:
    
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__ (self, maze, tick_length = 1, verbose = True):
        """
        Initializes the environment from a given maze, specified as an
        array of strings with maze elements
        :maze: The array of strings specifying the challenge
        :tick_length: The duration between agent decisions, in seconds
        :verbose: Whether or not the maze updates will be printed
        """
        self._maze = maze
        self._rows = len(maze)
        self._cols = len(maze[0])
        self._tick_length = tick_length
        self._verbose = verbose
        self._pellets = set()
        self._goals = set()
        self._walls = set()

        self._pellets_eaten = 0
        
        # Scan for pits and goals in the input maze
        for (row_num, row) in enumerate(maze):
            for (col_num, cell) in enumerate(row):
                if cell == Constants.WALL_BLOCK:
                    self._walls.add((col_num, row_num))
                if cell == Constants.GOAL_BLOCK:
                    self._goals.add((col_num, row_num))
                if cell == Constants.PELLET_BLOCK:
                    self._pellets.add((col_num, row_num))
                if cell == Constants.PLR_BLOCK:
                    self._player_loc = self._initial_loc = (col_num, row_num)
        self._spcl = self._pellets | self._goals | self._walls
        self._wrn1_tiles = self._get_wrn_set([self._get_adjacent(loc, 1) for loc in self._pellets])
        self._wrn2_tiles = self._get_wrn_set([self._get_adjacent(loc, 2) for loc in self._pellets])

        # Initialize the MazeAgent and ready simulation!
        self._goal_reached = False
        self._ag_maze = self._make_agent_maze()
        self._ag_tile = Constants.SAFE_BLOCK
        self._maze = [list(row) for row in maze] # Easier to change elements in this format
        self._og_maze = copy.deepcopy(self._maze)
        self._og_maze[self._player_loc[1]][self._player_loc[0]] = Constants.SAFE_BLOCK
        for (c, r) in self._wrn2_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_2
        for (c, r) in self._wrn1_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_1

        # Initialize MazeAgent here
        self._agent = PacmanAgent()
    
    
    ##################################################################
    # Methods
    ##################################################################
    
    def get_player_loc (self):
        """
        Returns the player's current location as a maze tuple
        """
        return self._player_loc
    
    def get_goal_loc (self):
        return next(iter(self._goals))
    
    def get_agent_maze (self):
        """
        Returns the agent's mental model of the maze, without key
        components revealed that have yet to be explored. Unknown
        spaces are filled with "?"
        """
        return self._ag_maze
    
    def start_mission (self):
        """
        Manages the agent's action loop and the environment's record-keeping
        mechanics
        """
        score = 0
        while (score > Constants.get_min_score()):
            time.sleep(self._tick_length)
            print("*** NEW TICK CYCLE ***")
            
            # Get player's next move in their plan, then execute
            next_act = self._agent.choose_action(self._ag_maze)
            print("next_act: ", next_act)
            self._move_request(next_act)
            
            # Return a perception for the agent to think about and plan next
            # perception = {"loc": self._player_loc, "tile": self._ag_tile}
            # self._agent.choose_action(perception)
            
            # Assess the post-move penalty and whether or not the game is complete
            # if self._pellet_test(self._player_loc):
            #     print("PELLET EATEN!!!")
            #     self._pellets_eaten += Constants.get_pellet_reward()
            # print("PELLET SCORE:", self._pellets_eaten)

            penalty = Constants.get_mov_penalty()
            score = score - penalty
            if self._verbose:
                print(
                    "\nCurrent Loc: " + str(self._player_loc) + " [" + self._ag_tile + "]" +
                    "\nLast Move: " + str(next_act) + 
                    "\nScore: " + str(score) +
                    "\nPellets Eaten: " + str(self._pellets_eaten) + 
                    "\n")
            if self._goal_test(self._player_loc):
                break
        
        if self._verbose:
            print("[!] Game Complete! Final Score: " + str(score))
        return score
            
    
    ##################################################################
    # "Private" Helper Methods
    ##################################################################

    def _get_adjacent (self, loc, offset):
        """
        Returns a set of the 4 adjacent cells to the given loc
        """
        (x, y) = loc
        pos_locs = [(x+offset, y), (x-offset, y), (x, y+offset), (x, y-offset)]
        return list(filter(lambda loc: loc[0] >= 0 and loc[1] >= 0 and loc[0] < self._cols and loc[1] < self._rows, pos_locs))
    
    def _get_wrn_set (self, wrn_list):
        return {item for sublist in wrn_list for item in sublist if item not in self._spcl}
    
    # Print the maze and agent's maze to the screen
    def _display (self):
        for (rowIndex, row) in enumerate(self._maze):
            print(''.join(row) + "\t" + ''.join(self._ag_maze[rowIndex]))
        
    def _wall_test (self, loc):
        return loc in self._walls
    
    def _goal_test (self, loc):
        return loc in self._goals
    
    def _pellet_test (self, loc):
        result = loc in self._pellets

        print("PELLETS SET BEFORE:")
        print(self._pellets)

        if result:
            self._pellets.remove(loc) # remove the pellet after being eaten

        print("PELLETS SET AFTER:")
        print(self._pellets)
        return result
        
    def _make_agent_maze (self):
        """
        Converts the 'true' maze into one with hidden tiles (?) for the agent
        to update as it learns
        """
        sub_regexp = "[" + Constants.PELLET_BLOCK + Constants.SAFE_BLOCK + "]"
        return [list(re.sub(sub_regexp, Constants.UNK_BLOCK, r)) for r in self._maze]
    
    def _move_request (self, move):
        old_loc = self._player_loc
        print("old_loc:", old_loc)

        new_loc = old_loc if move == None else tuple(sum(x) for x in zip(self._player_loc, Constants.MOVE_DIRS[move]))
        print("new_loc:", new_loc)

        # If you hit a wall, move back and do nothing
        if self._wall_test(new_loc):
            new_loc = old_loc
        else: # otherwise, process the new location
            if self._pellet_test(new_loc):
                print("PELLET EATEN!!!")
                self._pellets_eaten += Constants.get_pellet_reward()
                print("PELLET SCORE:", self._pellets_eaten)

                print("PELLET BEFORE:")
                self._display()
                print()

                # Remove the Pellet from the OG (original) maze
                self._og_maze[new_loc[1]][new_loc[0]] = Constants.SAFE_BLOCK

                print("PELLET AFTER:")
                self._display()
                print()

        self._update_mazes(self._player_loc, new_loc)
        self._player_loc = new_loc
        if self._verbose:
            self._display()
    
    def _update_mazes (self, old_loc, new_loc):
        # Actual Maze
        self._maze[old_loc[1]][old_loc[0]] = self._og_maze[old_loc[1]][old_loc[0]]
        self._maze[new_loc[1]][new_loc[0]] = Constants.PLR_BLOCK

        # Agent Perception Maze
        self._ag_maze[old_loc[1]][old_loc[0]] = self._og_maze[old_loc[1]][old_loc[0]]
        self._ag_maze[new_loc[1]][new_loc[0]] = Constants.PLR_BLOCK

        # Agent "tile"
        self._ag_tile = self._og_maze[new_loc[1]][new_loc[0]]

# Appears here to avoid circular dependency
from maze_agent import MazeAgent

if __name__ == "__main__":
    """
    Some example mazes with associated difficulties are
    listed below. The score thresholds given are for agents that actually use logic.
    Making a B-line for the goal on these mazes *may* satisfy the threshold listed here,
    but will not in general, more thorough tests.
    """
    mazes = [
        # Easy difficulty: Score > -20
        ["XXXXXX",
         "X...GX",
         "X..PPX",
         "X....X",
         "XP.P.X",
         "X@P..X",
         "XXXXXX"],
        
        # Medium difficulty: Score > -30
        ["XXXXXXXXX",
         "X..PGP..X",
         "X.......X",
         "X..P.P..X",
         "X.......X",
         "X..@....X",
         "XXXXXXXXX"],
        
        # Hard difficulty: Score > -35
        ["XXXXXXXXX",
         "X..PG...X",
         "X.......X",
         "X.P.P.P.X",
         "XP.....PX",
         "X...@...X",
         "XXXXXXXXX"],
    ]

    agent = PacmanAgent()
    
    # Pick your difficulty!
    env = Environment(mazes[0]) # Call with tick_length = 0 for instant games
    env.start_mission()
