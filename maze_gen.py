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
    def _get_new_maze (maze, n_ghosts, n_pellets, legal_positions):
        result = dict()
        positions = random.sample(legal_positions, n_ghosts + n_pellets + 1)
        result["positions"] = {
            "ghosts": positions[0:n_ghosts],
            "pellets": positions[n_ghosts:n_ghosts+n_pellets],
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
        base_maze = [re.sub(r"[PG@]+", ".", row) for row in maze]
        rows, cols = len(base_maze), len(base_maze[0])
        n_ghosts = n_pellets = 0
        legal_positions = []
        
        # Process original maze contents
        for r in range(rows):
            for c in range(cols):
                curr_cell = maze[r][c]
                if (curr_cell == "P"):
                    n_pellets += 1
                if (curr_cell == "G"):
                    n_ghosts += 1
                if (curr_cell != "X"):
                    legal_positions.append((c, r))
        
        training_mazes  = [MazeGen._get_new_maze(base_maze, n_ghosts, n_pellets, legal_positions) for _ in range(n_samples)]
        training_labels = [MazeGen._generate_label(m["maze"], m["positions"]) for m in training_mazes]
        training_mazes  = [m["maze"] for m in training_mazes]
        full_training   = {"X": training_mazes, "y": training_labels}
        return pd.DataFrame(data=full_training)

if __name__ == "__main__":
    maze = ["XXXXXXXXX",
            "X..P....X",
            "X.G..G..X",
            "X..XXXP.X",
            "XP.....PX",
            "X...@...X",
            "XXXXXXXXX"]
    
    result = MazeGen.get_labeled_data(maze, 100)
    print(result)
    