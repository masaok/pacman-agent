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

# Whether or not pellets are in new, different positions for each generated maze
RAND_PELLETS = False
N_SAMPLES = 10000

class MazeGen:
    
    # TODO: If not behaving well, issue might be how the label is being selected
    @staticmethod
    def _generate_label (maze, positions):
        mp = MazeProblem(maze)
        pac_pos = positions["pacman"][0]
        best_cost, best_act = float('inf'), None
        
        for pellet_pos in positions["pellets"]:
            cost, path = pathfind(mp, pac_pos, pellet_pos)
            if cost < best_cost:
                best_cost, best_act = cost, path[0]
        
        return best_act

    @staticmethod
    def _get_new_maze (maze, n_ghosts, n_pellets, legal_positions, pellet_positions, rand_pellets):
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
        
        training_mazes  = [MazeGen._get_new_maze(base_maze, n_ghosts, n_pellets, legal_positions, pellet_positions, RAND_PELLETS) for _ in range(n_samples)]
        training_labels = [MazeGen._generate_label(m["maze"], m["positions"]) for m in training_mazes]
        training_mazes  = [m["maze"] for m in training_mazes]
        full_training   = {"X": training_mazes, "y": training_labels}
        return pd.DataFrame(data=full_training)

if __name__ == "__main__":
    maze = ["XXXXXXXXX",
            "X..O....X",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X...P...X",
            "XXXXXXXXX"]
    
    pd.set_option('display.max_rows', None)
    result = MazeGen.get_labeled_data(maze, N_SAMPLES)
    result["X"] = result["X"].transform(lambda x: "\n".join(x))
    result.to_csv("./dat/generated_data.csv", index=False)
    
    