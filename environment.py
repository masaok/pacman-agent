'''
Environment class responsible for configuring and running the maze
'''

import copy
import time

import tkinter as tk
from tkinter import *

from constants import Constants
from ghost_agent import GhostAgent
from pacman_agent import PacmanAgent
from maze_ui import MazeUI
from maze_problem import MazeProblem

# from PIL import Image, ImageTk

import argparse


class Environment:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze, window, tick_length=1, verbose=True, debug=True, step=True):
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
        self._step = step

        # Maze block sets
        self._ghosts = set()
        self._pellets = set()
        self._walls = set()

        # Score keeping
        self._score = 0
        self._pellets_eaten = 0

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

        self._spcl = self._pellets | self._walls
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

        # Initialize MazeAgent here
        self._agent = PacmanAgent(maze)
        self._index = 0  # keep track of which loop we're on

        # Graphics Test

        # window.geometry("600x600")
        # canvas = Canvas(window, width=500, height=300)
        # # canvas.pack()

        # self.filename = "ui/images/red_ghost_trans.png"
        # # img = Image.open(self.filename)
        # img = PhotoImage(file=self.filename)
        # canvas.create_image(0, 0, image=img)
        # # img = ImageTk.PhotoImage(img)

        # # canvas.create_image(0, 0, anchor=NW, image=img)
        # canvas.create_rectangle(0, 0, 80, 80, fill="purple")
        # # canvas.create_image(0, 0, anchor=NW, image=img)
        # # canvas.create_rectangle(100, 100, 80, 80, fill="pink")
        # # canvas.create_rectangle(300, 300, 80, 80, fill="green")

        # window.update()

        # Graphics GUI Stuff (tk)
        self._window = window
        self._maze_ui = MazeUI(self._window, self._maze)

    ##################################################################
    # Methods
    ##################################################################

    def get_player_loc(self):
        """
        Returns the player's current location as a maze tuple
        """
        return self._player_loc

    def move(self):

        # Draw the Maze first, before any movement
        self._maze_ui.draw_maze()
        if self._step:
            # Wait for the button to be pressed to step forward
            self._maze_ui.btn.wait_variable(self._maze_ui.btn_var)

        print("index: ", self._index)
        print("*** NEW TICK CYCLE ***")

        print("GHOSTS:")
        print(self._ghosts)

        self._index += 1

        # Get player's next move in their plan, then execute
        mp = MazeProblem(self._maze)
        next_act = self._agent.choose_action(self._maze, mp.legal_actions(self.get_player_loc()))
        print("next_act: ", next_act)
        self._move_request(next_act)

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
            print(self._player_loc)
            self._insert_block(self._player_loc, Constants.DEATH_BLOCK)
            self._cleanup()
            return

        print("GHOSTS MOVING ...")
        print(self._ghosts)

        self._move_ghosts()
        self._display()
        print("\n")
        if self._ghost_test(self._player_loc):
            print("GHOSTS MOVED AND HIT PACMAN!  DONE!")
            print(self._player_loc)
            self._insert_block(self._player_loc, Constants.DEATH_BLOCK)
            self._cleanup()
            return

        # if self._verbose:
        #     print("[!] Game Complete! Final Score: " + str(self._score))

        # # MazeUI Stuff

        if self._step:
            # Wait for the button to be pressed to step forward
            # self._maze_ui.btn.wait_variable(self._maze_ui.btn_var)
            self._window.after(0, self.move)
        else:
            self._window.after(self._tick_length * 1000, self.move)

    ##################################################################
    # "Private" Helper Methods
    ##################################################################
    def _cleanup(self):
        if not self._step:
            # One more sleep before death if in animation mode
            time.sleep(self._tick_length)
        self._maze_ui.draw_maze()  # Draw the final maze

    def _get_adjacent(self, loc, offset):
        """
        Returns a set of the 4 adjacent cells to the given loc
        """
        (x, y) = loc
        pos_locs = [(x+offset, y), (x-offset, y), (x, y+offset), (x, y-offset)]
        return list(
            filter(
                lambda loc: loc[0] >= 0 and loc[1] >= 0 and loc[0] < self._cols and loc[1] <
                self._rows, pos_locs))

    def _get_wrn_set(self, wrn_list):
        return {item for sublist in wrn_list for item in sublist if item not in self._spcl}

    # Print the maze and agent's maze to the screen
    def _display(self):
        for (rowIndex, row) in enumerate(self._maze):
            # print(''.join(row) + "\t" + ''.join(self._maze[rowIndex]))
            print(''.join(self._maze[rowIndex]))

    def _wall_test(self, loc):
        return loc in self._walls

    def _ghost_test(self, loc):
        return loc in self._ghosts

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

    def _insert_block(self, location, block):
        self._maze[location[1]][location[0]] = block

    def _move_ghosts(self):

        index = 0
        for ghost in self._ghosts.copy():

            # TODO: create new function for actions for ghosts
            ghost_agent = GhostAgent(ghost)
            move = ghost_agent.choose_action(self._maze, self._player_loc)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pacman Trainer Deep Learning GUI')
    parser.add_argument(
        '--animate', '-a',
        action='store_true',
        help='Animate (disable stepping through the animation)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        default=False,
        help='Show debug output'
    )
    parser.add_argument(
        '--step', '-s',
        action='store_true',
        help='Step through each cycle of animation (default)'
    )
    args = parser.parse_args()
    print("args: " + str(args))

    if args.animate:
        args.step = False

    """
    Some example mazes with associated difficulties are
    listed below. The score thresholds given are for agents that actually use logic.
    Making a B-line for the goal on these mazes *may* satisfy the threshold listed here,
    but will not in general, more thorough tests.
    """
    maze = ["XXXXXXXXX",
            "X..O....X",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X...P...X",
            "XXXXXXXXX"]

    # Exit the Python app cleanly in terminal
    # Credit: https://stackoverflow.com/q/69917376/10415969
    def on_exit():
        env._maze_ui.window.destroy()  # env window
        env._maze_ui.btn_var.set("")
        exit(0)

    window = tk.Tk()
    window.protocol('WM_DELETE_WINDOW', on_exit)

    # Add a window title
    # https://pythonguides.com/python-tkinter-title/
    # https://stackoverflow.com/questions/2395431/using-tkinter-in-python-to-edit-the-title-bar
    window.title("Pacman Imitation Learning")

    # Start the environment
    # Call with tick_length = 0 for instant games
    # TODO: Add command-line options so debug can be passed as a flag
    step = True
    debug = False
    env = Environment(maze, window, debug=args.debug, step=args.step)

    # Graphical
    env.move()
    print("END MAIN MOVE")

    window.mainloop()
    print("END MAIN LOOP")

    # Command-line
    # env.start_mission()
