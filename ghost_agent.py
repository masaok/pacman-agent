import random
from constants import *
from maze_problem import *
from pathfinder import *
from queue import Queue
from pprint import pprint


class GhostAgent:
    def __init__(self, location):
        self.loc = location

    def choose_action(self, maze, player_location):
        """
        Returns an action from a set {U, D, L, R} given perception (maze)
        10% of the time returns a random choice
        90% of the time moves closer to Pacman

        Parameters:
            maze (array): array of strings representation of the maze

        Returns:
            str: direction to move
        """

        will_chase = random.uniform(0, 1) > Constants.GHOST_EPSILON

        # Chases Pacman by plotting the next best move to close the distance using A* search
        # if the given roll was greater than the ghost's epsilon chance
        if will_chase:
            mp = MazeProblem(maze)
            pac_loc = mp.get_player_loc()
            best_path = pathfind(mp, self.loc, pac_loc)
            return best_path[1][0]

        # Otherwise, move in a random direction
        else:
            return random.sample(Constants.MOVES, 1)[0]
