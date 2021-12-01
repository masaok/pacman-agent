'''
Faux training data generator for Pacman Trainer
'''

import time
import random
import re
import pandas as pd
from maze_problem import MazeProblem
from pathfinder import *
from constants import Constants

class MazeGen:
    """
    Maze Generation class used to create faux-samples for the Pacman Trainer
    """
    
    @staticmethod
    def _generate_label (maze, positions):
        """
        Assigns a faux-label to the given maze that is the action which brings
        Pacman closest to the nearest pellet
        """
        mp = MazeProblem(maze)
        pac_pos = positions["pacman"][0]
        best_cost, best_act = float('inf'), None
        
        # TODO: Can be improved to account for ghosts, this is just for illustrative
        # purposes!
        for pellet_pos in positions["pellets"]:
            cost, path = pathfind(mp, pac_pos, pellet_pos)
            if cost < best_cost:
                best_cost, best_act = cost, path[0]
        
        return best_act

    @staticmethod
    def _get_new_maze (maze, n_ghosts, n_pellets, legal_positions, pellet_positions, rand_pellets):
        """
        Generates a new maze with the walls in fixed positions, and the number of pellets / ghosts fixed,
        but in random* positions.
        * If rand_pellets is True, fixes the positions of the pellets to the given pellet_positions
        """
        result = dict()
        positions = random.sample(legal_positions, n_ghosts + 1 if not rand_pellets else n_ghosts + n_pellets + 1)
        result["positions"] = {
            "ghosts": positions[0:n_ghosts],
            "pellets": random.sample(pellet_positions if not rand_pellets else positions[n_ghosts:n_ghosts+n_pellets], random.sample(range(1,n_pellets+1), 1)[0]),
            "pacman": positions[-1:]
        }
        new_maze = maze.copy()
        
        rep_chars = {"ghosts": Constants.GHOST_BLOCK, "pellets": Constants.PELLET_BLOCK, "pacman": Constants.PLR_BLOCK}
        for entity in result["positions"].keys():
            for pos in result["positions"][entity]:
                c, r = pos
                new_maze[r] = new_maze[r][:c] + rep_chars[entity] + new_maze[r][c+1:]
        result["maze"] = new_maze
        return result

    @staticmethod
    def get_labeled_data (maze, n_samples):
        """
        Generates n_samples number of maze combinations from the given base-maze in which
        the walls are in fixed position, but the other maze contents are randomized (except
        when some generation parameters are set, like Constants.RAND_PELLETS).
        """
        # Create base maze with only the walls remaining
        base_maze = [re.sub(r"[" + Constants.PELLET_BLOCK + Constants.PLR_BLOCK + Constants.GHOST_BLOCK + "]", ".", row) for row in maze]
        rows, cols = len(base_maze), len(base_maze[0])
        n_ghosts = n_pellets = 0
        legal_positions = []
        pellet_positions = []
        
        # Process original maze contents
        for r in range(rows):
            for c in range(cols):
                curr_cell = maze[r][c]
                if (curr_cell == Constants.PELLET_BLOCK):
                    n_pellets += 1
                    pellet_positions.append((c, r))
                if (curr_cell == Constants.GHOST_BLOCK):
                    n_ghosts += 1
                if (curr_cell != Constants.WALL_BLOCK and curr_cell != Constants.PELLET_BLOCK):
                    legal_positions.append((c, r))
        
        training_mazes  = [MazeGen._get_new_maze(base_maze, n_ghosts, n_pellets, legal_positions, pellet_positions, Constants.RAND_PELLETS) for _ in range(n_samples)]
        training_labels = [MazeGen._generate_label(m["maze"], m["positions"]) for m in training_mazes]
        training_mazes  = [m["maze"] for m in training_mazes]
        full_training   = {"X": training_mazes, "y": training_labels}
        return pd.DataFrame(data=full_training)

if __name__ == "__main__":
    """
    Used to generate csv output for mazes
    """
    print("[!] Beginning Maze Generation")
    result = MazeGen.get_labeled_data(Constants.MAZE, Constants.N_SAMPLES)
    result["X"] = result["X"].transform(lambda x: "\n".join(x))
    result.to_csv(Constants.MAZE_GEN_PATH, index=False)
    print("[!] Maze Generation Completed!")
    
