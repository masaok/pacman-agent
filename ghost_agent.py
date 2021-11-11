import random
from pathfinder import *
from queue import Queue
from pprint import pprint


class GhostAgent:
    def __init__(self, location):
        self.location = location

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

        rand = random.uniform(0, 1)
        method = 'random'
        if (rand > 0.1):  # 90% of the time go with distance calc and move closer
            method = 'distance'
        direction = ''

        print("GHOST CHOOSE ACTION maze:")
        pprint(maze)

        print("GHOST LOCATION: ", self.location)
        print("PLAYER LOCATION: ", player_location)
        print("PLAYER LOCATION x: ", player_location[0])
        print("PLAYER LOCATION y: ", player_location[1])

        print("GHOST MOVE METHOD: ", method)

        # TODO: Is this Manhattan distance?  (it's faster than testing each possible move)
        if method == 'distance':
            # Determine whether X or Y distance is smaller
            move_axis = 0  # default to x
            if (self.location[0] == player_location[0]):  # If same X axis ...
                print("SAME X, MOVING ON Y ...")
                move_axis = 1  # Move on Y
            elif (self.location[1] == player_location[1]):  # If same Y axis
                print("SAME Y, MOVING ON X ...")
                move_axis = 0  # Move on X
            elif (abs(self.location[0] - player_location[0]) > abs(self.location[1] - player_location[1])):
                print("GHOST IS CLOSER ON Y AXIS, MOVING ON Y ...")
                move_axis = 1

            # Given the smaller axis diff, determine the direction to move on that axis
            if move_axis == 0:  # If using X axis ...
                print("GHOST MOVE X AXIS ...")
                if self.location[0] - player_location[0] > 0:
                    direction = 'L'  # Move left if ghost is to the right of player
                else:
                    direction = 'R'
            else:  # Use the Y axis
                print("GHOST MOVE Y AXIS ...")
                if self.location[1] - player_location[1] > 0:
                    direction = 'U'  # Move up if ghost below player
                else:
                    direction = 'D'

        else:
            directions = ["U", "D", "L", "R"]
            random_index = random.randint(0, len(directions)-1)
            direction = directions[random_index]

        print("GHOST MOVE DIRECTION: ", direction)

        return direction
