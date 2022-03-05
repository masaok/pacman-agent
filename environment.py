'''
Environment class responsible for configuring and running the maze
'''

import copy
import time

import tkinter as tk
import random
from tkinter import *
from constants import Constants
from ghost_agent import GhostAgent
from pacman_agent import PacmanAgent
from maze_ui import MazeUI
from maze_problem import MazeProblem

# from PIL import Image, ImageTk

import argparse


class Environment:
    
    running_env = None

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze, verbose=False, debug=False, step=False, gui=True):
        """
        Initializes the environment from a given maze, specified as an
        array of strings with maze elements
        :maze: The array of strings specifying the challenge
        :tick_length: The duration between agent decisions, in seconds
        :verbose: Whether or not the maze updates will be printed
        """
        self._maze = maze
        # Easier to change elements in this format
        self._maze = [list(row) for row in maze]
        self._mp = MazeProblem(maze)
        self._rows = len(maze)
        self._cols = len(maze[0])
        self._tick_length = Constants.TICK_LEN / 1000.0
        self._debug = debug
        self._verbose = verbose
        self._step = step
        self._moves = 0

        # Maze block sets
        self._ghosts = self._mp.get_ghosts()
        self._pellets = self._mp.get_pellets()
        self._walls = self._mp.get_walls()
        self._open = self._mp.get_open()
        self._player_loc = self._initial_loc = self._mp.get_player_loc()
        
        # Random start state behavior
        # self._player_loc = random.choice(list(self._open))
        

        # Score keeping
        self._pellets_eaten = 0
        self._n_pellets = len(self._pellets)
        self._spcl = self._pellets | self._walls

        # Initialize the MazeAgent and ready simulation!
        self._goal_reached = False

        # Keep a copy of the original maze
        self._og_maze = copy.deepcopy(self._maze)
        self._og_maze[self._player_loc[1]][self._player_loc[0]] = Constants.SAFE_BLOCK

        # Initialize MazeAgent here
        self._agent = PacmanAgent(maze)
        self._index = 0  # keep track of which loop we're on
        self._last_act = None

        # Graphics GUI Stuff (tk)
        self._gui = gui
        if gui:
            window = tk.Tk()
            # window.protocol('WM_DELETE_WINDOW', on_exit)
        
            # Add a window title
            # https://pythonguides.com/python-tkinter-title/
            # https://stackoverflow.com/questions/2395431/using-tkinter-in-python-to-edit-the-title-bar
            window.title("Pacman Reinforcement Learning")
            self._window = window
            self._maze_ui = MazeUI(self._window, self._maze, self._debug)
        
        self.outcome = {"moves": self._moves, "win": False, "pellets": 0, "max_pellets": self._n_pellets}

    ##################################################################
    # Methods
    ##################################################################

    def get_player_loc(self):
        """
        Returns the player's current location as a maze tuple
        """
        return copy.deepcopy(self._player_loc)

    def move(self):
        state = copy.deepcopy(self._maze)

        self._moves += 1
        if self._moves > Constants.MAX_MOVES:
            print("TIME's UP!  GAME OVER!")
            if self._debug:
                print(self._player_loc)
            self._insert_block(self._player_loc, Constants.TIMEOUT_BLOCK)
            next_state = copy.deepcopy(self._maze)
            self._agent.give_transition(state, self._last_act, next_state, True)
            self._cleanup()
            return
        
        # Draw the Maze first, before any movement
        if self._gui:
            self._maze_ui.draw_maze()
        if self._step and self._gui:
            # Wait for the button to be pressed to step forward
            self._maze_ui.btn.wait_variable(self._maze_ui.btn_var)

        if self._debug:
            print("index: ", self._index)
            print("*** NEW TICK CYCLE ***")
    
            print("GHOSTS:")
            print(self._ghosts)

        self._index += 1

        # Get player's next move in their plan, then execute
        mp = MazeProblem(self._maze)
        next_act = self._agent.choose_action(self._maze, mp.legal_actions(self.get_player_loc()))
        self._last_act = next_act
        if self._debug:
            print("next_act: ", next_act)
        self._move_request(next_act)

        if self._verbose:
            print(
                "\nCurrent Loc: " + str(self._player_loc) +
                "\nLast Move: " + str(next_act) +
                "\nPellets Eaten: " + str(self._pellets_eaten) +
                "\nMoves: " + str(self._moves) + " / " + str(Constants.MAX_MOVES) +
                "\n")

        # Terminal State Checks
        if self._ghost_test(self._player_loc):
            print("PACMAN MOVED AND HIT A GHOST!  GAME OVER!")
            if self._debug:
                print(self._player_loc)
            self._insert_block(self._player_loc, Constants.DEATH_BLOCK)
            next_state = copy.deepcopy(self._maze)
            self._agent.give_transition(state, next_act, next_state, True)
            self._cleanup()
            return
        
        if self._pellets_eaten == self._n_pellets:
            print("ALL PELLETS EATEN! HUZZAH YOU GLUTTON!  GAME OVER!")
            self._insert_block(self._player_loc, Constants.WIN_BLOCK)
            next_state = copy.deepcopy(self._maze)
            self._agent.give_transition(state, next_act, next_state, True)
            self.outcome["win"] = True
            self._cleanup()
            return

        if self._debug:
            print("GHOSTS MOVING ...")
            print(self._ghosts)

        # Ghost Move Turn
        self._move_ghosts()
        if self._verbose:
            self._display()
            print("\n")
            
        if self._ghost_test(self._player_loc):
            print("GHOSTS MOVED AND HIT PACMAN!  GAME OVER!")
            if self._debug:
                print(self._player_loc)
            self._insert_block(self._player_loc, Constants.DEATH_BLOCK)
            next_state = copy.deepcopy(self._maze)
            self._agent.give_transition(state, next_act, next_state, True)
            self._cleanup()
            return

        next_state = copy.deepcopy(self._maze)
        self._agent.give_transition(state, next_act, next_state, False)

        # MazeUI Stuff

        if self._step:
            # Wait for the button to be pressed to step forward
            self._window.after(0, self.move)
        else:
            if self._gui:
                self._window.after(Constants.TICK_LEN, self.move)
            else:
                time.sleep(self._tick_length)
                self.move()


    ##################################################################
    # "Private" Helper Methods
    ##################################################################
    
    def _cleanup(self):
        if not self._step:
            # One more sleep before death if in animation mode
            time.sleep(self._tick_length)
        self.outcome["pellets"] = self._pellets_eaten
        self.outcome["moves"] = self._moves
        self._agent.give_terminal()
        if self._gui:
            self._maze_ui.draw_maze()  # Draw the final maze
            self._maze_ui.destroy()

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

        new_loc = old_loc if move == None else tuple(
            sum(x) for x in zip(self._player_loc, Constants.MOVE_DIRS[move]))
        if self._verbose:
            print("old_loc:", old_loc)
            print("new_loc:", new_loc)

        # If you hit a wall, move back and do nothing
        if self._wall_test(new_loc):
            new_loc = old_loc
        else:  # otherwise, process the new location
            if self._pellet_test(new_loc):
                self._pellets_eaten += 1
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
            
    def run_game (verbose=False, debug=False, step=False, gui=True):
        # Start the environment
        # Call with tick_length = 0 for instant games
        Environment.running_env = Environment(Constants.MAZE, verbose=verbose, debug=debug, step=step, gui=gui)
    
        # Graphical
        Environment.running_env.move()
    
        if gui:
            Environment.running_env._window.mainloop()
        
        return Environment.running_env.outcome


# Exit the Python app cleanly in terminal
# Credit: https://stackoverflow.com/q/69917376/10415969
def on_exit():
    if not Environment.running_env is None:
        Environment.running_env._maze_ui.window.destroy()  # env window
        Environment.running_env._maze_ui.btn_var.set("")
    # exit(0)

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

    Environment.run_game(False, False)

