# pacman-agent

## Prerequisites

- Python 3 (so far, it works with Python 3, but if there are problems, try Python 2)

## Environments Tested

### Windows 10 (Build 19042.1288)

- Git Bash
- Python 3.9.8

## Quickstart

```
$ python3 -m venv env
$ source env/Script/activate  # Windows Git Bash only
$ source env/bin/activate     # Mac / Linux only
$ pip install -r requirements.txt
$ python3 -u environment.py
```

```
$ deactivate  # exit virtual environment
```

## Components

- **PacmanAgent**: Contains all of the logic for the agent controlling Pacman
- **Environment**: Contains the game logic and is largely structured in the same way as the Wumpus World / Blindbot’s environment
  - A constructor, parameterized by the Pacman Trainer’s array-of-strings maze representation, that initializes all of the maze variables, as follows:
    - The maze structure, where walls are located, and thus what legal moves are available in any position.
    - The sets and positions of all game actors (beginning with their provided starting positions), including:
      - Pacman
      - Ghosts
      - Pellets
  - Pacman’s score (# of pellets eaten)
  - Determining game-ending conditions (viz., Pacman getting eaten by a ghost or eating every pellet)
  - Rendering the maze elements (in their basic array-of-strings format like in the Pacman Trainer) in terms of their graphical equivalents
  - Actors in the environment take turns acting, so the environment determines what happens on every “tick” or turn of a running game:
    - The PacmanAgent’s chooseAction method is called with the current game state, and its action choice is enacted (if it’s not a legal action, i.e., one that runs it into a wall, it does nothing that turn).
      - If Pacman eats a pellet, it’s removed from the board and the score is incremented
      - If Pacman moves onto a ghost’s tile, it dies and the game’s over
- **MazeUI**: Given a Tk "root window", draw a Tk Canvas on it and all the blocks, characters, and items that make up a maze

## TODO

- All ghosts make an action choice governed by a coin-flip: 10% of the time, it will choose randomly, and the other 90%, it will take a step that brings it closer to Pacman. You can use / adapt the Pathfinder class I gave in the BlindBot package to help with pathfinding or just do something basic like looking at the Manhattan distance between a ghost and Pacman and then choosing the action that minimizes it.
- Add a command line switch to run in "debug" mode

## Reminders

- There will only be walls, pellets, ghosts, and open squares in the final version; the remnants of the BlindBot's "end goal" and "pits" will not be in the final version.
- There won't be 2 versions of the maze state like we had with BlindBot (which were the agent's ag_maze and the environment's maze) since this will be a perfect-information game (the whole board is always visible to the agent, unlike for BlindBot).
- The agent's choose_action method is called with the current perception (the maze state), which returns an action, which is really all we need to accomplish the "black box" part of the imitation learner.
