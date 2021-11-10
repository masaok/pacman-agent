'''
Environment class responsible for configuring and running
the MazePitfall problem with BlindBot agent
'''


import os
import re
import sys
import time
import copy

import tkinter as tk

from constants import Constants
from maze_agent import MazeAgent
from pacman_agent import PacmanAgent
from maze_ui import MazeUI


class Environment:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze, window, tick_length=1, verbose=True, debug=True):
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
        self._debug = debug
        self._verbose = verbose

        # Maze block sets
        self._ghosts = set()
        self._goals = set()
        self._pellets = set()
        self._walls = set()

        # Score keeping
        self._score = 0
        self._pellets_eaten = 0

        # Scan for pits and goals in the input maze
        for (row_num, row) in enumerate(maze):
            for (col_num, cell) in enumerate(row):
                if cell == Constants.WALL_BLOCK:
                    self._walls.add((col_num, row_num))
                if cell == Constants.END_BLOCK:
                    self._goals.add((col_num, row_num))
                if cell == Constants.GHOST_BLOCK:
                    self._ghosts.add((col_num, row_num))
                if cell == Constants.PELLET_BLOCK:
                    self._pellets.add((col_num, row_num))
                if cell == Constants.PLR_BLOCK:
                    self._player_loc = self._initial_loc = (col_num, row_num)

        self._spcl = self._pellets | self._goals | self._walls
        self._wrn1_tiles = self._get_wrn_set(
            [self._get_adjacent(loc, 1) for loc in self._pellets])
        self._wrn2_tiles = self._get_wrn_set(
            [self._get_adjacent(loc, 2) for loc in self._pellets])

        # Initialize the MazeAgent and ready simulation!
        self._goal_reached = False
        # Easier to change elements in this format
        self._maze = [list(row) for row in maze]

        # Keep a copy of the original maze
        self._og_maze = copy.deepcopy(self._maze)
        self._og_maze[self._player_loc[1]
                      ][self._player_loc[0]] = Constants.SAFE_BLOCK

        for (c, r) in self._wrn2_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_2
        for (c, r) in self._wrn1_tiles:
            self._og_maze[r][c] = Constants.WRN_BLOCK_1

        # Initialize MazeAgent here
        self._agent = PacmanAgent()

        self._index = 0

        # self._maze_ui = MazeUI(maze)

        # Graphics GUI Stuff (tk)

        self._window = window
        self._maze_ui = MazeUI(self._window, self._maze)
        # self._window = tk.Tk()
        # self._window.geometry("500x500")

        # self._canvas = tk.Canvas(self._window, height=500, width=500)
        # self._canvas.grid(row=0, column=0, sticky='w')

        # coord = [x, y, x+width, y+height]
        # self._circle = self._canvas.create_oval(
        #     coord, outline="red", fill="red")

        # coord = [x, y, x+40, y+40]
        # self._rect2 = self._canvas.create_rectangle(
        #     coord, outline="Blue", fill="Blue")

        # self._move()

        # self._window.mainloop()

    ##################################################################
    # Methods
    ##################################################################

    def get_player_loc(self):
        """
        Returns the player's current location as a maze tuple
        """
        return self._player_loc

    def get_goal_loc(self):
        return next(iter(self._goals))

    def move(self):

        print("index: ", self._index)
        print("*** NEW TICK CYCLE ***")

        print("GHOSTS:")
        print(self._ghosts)

        # if (self._index >= 1):  # run after 1 second
        #     self._canvas.delete('all')

        # if (self._index >= 2):
        #     self._window.destroy()

        self._index += 1

        # Get player's next move in their plan, then execute
        next_act = self._agent.choose_action(self._maze)
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
        self._score -= penalty
        if self._verbose:
            print(
                # "\nCurrent Loc: " + str(self._player_loc) + " [" + self._ag_tile + "]" +
                "\nCurrent Loc: " + str(self._player_loc) +
                "\nLast Move: " + str(next_act) +
                "\nScore: " + str(self._score) +
                "\nPellets Eaten: " + str(self._pellets_eaten) +
                "\n")

        if self._ghost_test(self._player_loc):
            print("PACMAN MOVED AND HIT A GHOST!  DONE!")
            # self._window.destroy()
            return

        if self._goal_test(self._player_loc):
            print("PACMAN HIT THE GOAL!  DONE!")
            # self._window.destroy()
            return

        print("GHOSTS MOVING ...")
        self._move_ghosts()
        self._display()
        print("\n")
        if self._ghost_test(self._player_loc):
            print("GHOSTS MOVED AND HIT PACMAN!  DONE!")
            # self._window.destroy()
            return

        # if self._verbose:
        #     print("[!] Game Complete! Final Score: " + str(self._score))

        # Test MazeUI
        self._maze_ui.draw_maze()

        input("Press Enter to continue...")

        self._window.after(self._tick_length * 1000, self.move)

    ##################################################################
    # "Private" Helper Methods
    ##################################################################

    def _get_adjacent(self, loc, offset):
        """
        Returns a set of the 4 adjacent cells to the given loc
        """
        (x, y) = loc
        pos_locs = [(x+offset, y), (x-offset, y), (x, y+offset), (x, y-offset)]
        return list(filter(lambda loc: loc[0] >= 0 and loc[1] >= 0 and loc[0] < self._cols and loc[1] < self._rows, pos_locs))

    def _get_wrn_set(self, wrn_list):
        return {item for sublist in wrn_list for item in sublist if item not in self._spcl}

    # Print the maze and agent's maze to the screen
    def _display(self):
        for (rowIndex, row) in enumerate(self._maze):
            print(''.join(row) + "\t" + ''.join(self._maze[rowIndex]))

    def _wall_test(self, loc):
        return loc in self._walls

    def _ghost_test(self, loc):
        return loc in self._ghosts

    def _goal_test(self, loc):
        return loc in self._goals

    def _pellet_test(self, loc):
        result = loc in self._pellets

        if self._debug:
            print("PELLETS SET BEFORE:")
            print(self._pellets)

        if result:
            self._pellets.remove(loc)  # remove the pellet after being eaten

        if self._debug:
            print("PELLETS SET AFTER:")
            print(self._pellets)
        return result

    def _make_agent_maze(self):
        """
        Converts the 'true' maze into one with hidden tiles (?) for the agent
        to update as it learns
        """
        sub_regexp = "[" + Constants.PELLET_BLOCK + Constants.SAFE_BLOCK + "]"
        return [list(re.sub(sub_regexp, Constants.UNK_BLOCK, r)) for r in self._maze]

    def _move_request(self, move):
        old_loc = self._player_loc
        print("old_loc:", old_loc)

        new_loc = old_loc if move == None else tuple(
            sum(x) for x in zip(self._player_loc, Constants.MOVE_DIRS[move]))
        print("new_loc:", new_loc)

        # If you hit a wall, move back and do nothing
        if self._wall_test(new_loc):
            new_loc = old_loc
        else:  # otherwise, process the new location
            if self._pellet_test(new_loc):
                self._pellets_eaten += Constants.get_pellet_reward()
                if self._debug:
                    print("PELLET EATEN!!!")
                    print("PELLET SCORE:", self._pellets_eaten)

                # Remove the Pellet from the OG (original) maze
                self._og_maze[new_loc[1]][new_loc[0]] = Constants.SAFE_BLOCK

        self._update_mazes(self._player_loc, new_loc)
        self._player_loc = new_loc
        if self._verbose:
            self._display()

    def _update_mazes(self, old_loc, new_loc):
        # Actual Maze
        self._maze[old_loc[1]][old_loc[0]
                               ] = self._og_maze[old_loc[1]][old_loc[0]]
        self._maze[new_loc[1]][new_loc[0]] = Constants.PLR_BLOCK

        # Agent "tile" -- what is an agent tile?
        # self._ag_tile = self._og_maze[new_loc[1]][new_loc[0]]

    def _move_ghosts(self):

        index = 0
        for ghost in self._ghosts.copy():

            # TODO: create new function for actions for ghosts
            move = self._agent.choose_action(self._maze)

            if self._debug:
                print("GHOST " + str(index) + ":")
                print(ghost)

            old_loc = ghost
            new_loc = old_loc if move == None else tuple(
                sum(x) for x in zip(ghost, Constants.MOVE_DIRS[move]))

            if self._debug:
                print("  MOVING " + str(index) + " to " + str(move) + ":")
                print(new_loc)

            # If ghost hits a wall, move it back and do nothing
            if self._wall_test(new_loc):
                new_loc = old_loc
            elif self._ghost_test(new_loc):
                new_loc = old_loc
            # else: # otherwise, process the new location

            if self._debug:
                print("GHOSTS BEFORE REMOVAL:")
                print(self._ghosts)

            # Update Ghosts set
            self._ghosts.remove(old_loc)

            if self._debug:
                print("GHOSTS AFTER REMOVAL:")
                print(self._ghosts)

            self._ghosts.add(new_loc)

            if self._debug:
                print("GHOSTS AFTER ADD:")
                print(self._ghosts)

            # # Update the Mazes

            # If the old location was an OG Pellet *and* the Pellet still exists...
            # ...keep the pellet in the maze, otherwise, it's a safe block
            if self._og_maze[old_loc[1]][old_loc[0]] == Constants.PELLET_BLOCK and old_loc in self._pellets:
                self._maze[old_loc[1]][old_loc[0]] = Constants.PELLET_BLOCK
            else:
                self._maze[old_loc[1]][old_loc[0]] = Constants.SAFE_BLOCK

            # New location is a Ghost
            self._maze[new_loc[1]][new_loc[0]] = Constants.GHOST_BLOCK

            index += 1


# Appears here to avoid circular dependency

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
         "X.P..X",
         "XPGP.X",
         "X.P..X",
         "X....X",
         "X@...X",
         "XXXXXX"],

        # Medium difficulty: Score > -30
        ["XXXXXXXXX",
         "X..PEP..X",
         "X.......X",
         "X..P.P..X",
         "X.......X",
         "X..@....X",
         "XXXXXXXXX"],

        # Hard difficulty: Score > -35
        ["XXXXXXXXX",
         "X..PE...X",
         "X.......X",
         "X.P.P.P.X",
         "XP.....PX",
         "X...@...X",
         "XXXXXXXXX"],
    ]

    agent = PacmanAgent()

    window = tk.Tk()

    # Add a window title
    # https://pythonguides.com/python-tkinter-title/
    # https://stackoverflow.com/questions/2395431/using-tkinter-in-python-to-edit-the-title-bar
    window.title("Pacman Imitation Learning")

    # Start the environment
    # Call with tick_length = 0 for instant games
    env = Environment(mazes[0], window, debug=False)

    # Graphical
    env.move()
    window.mainloop()

    # Command-line
    # env.start_mission()
