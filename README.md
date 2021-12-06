# pacman-agent

## Prerequisites

- Python 3 (so far, it works with Python 3, but if there are problems, try Python 2)

## Environments Tested

### Windows 10 (Build 19042.1288)

- Git Bash
- Python 3.9.8

### MacOS 11.6 (Build 20G165)

- Python 3.9.4

## Quickstart

First, check out this repo and `cd` into the repo directory, then follow these steps:

```
$ python3 -m venv env
$ source env/Scripts/activate  # Windows Git Bash only
$ source env/bin/activate     # Mac / Linux only
$ pip install -r requirements.txt
$ python3 -u environment.py -h  # Show command-line help
$ python3 -u environment.py     # Run the app and show the GUI
```

```
$ deactivate  # exit virtual environment
```
### Windows WSL2 Ubuntu Extras

```
sudo apt update
sudo apt-get install python3-tk
```

#### VcXsrv (X server for Windows to run GUI stuff)

Download and install VcXsrv from here: https://sourceforge.net/projects/vcxsrv/

When running it (it's called XLaunch in Windows), make sure to check this box:

<img src="https://user-images.githubusercontent.com/1320083/144791402-763bdb42-94a4-490b-8d90-4cd898f55984.png" width="40%" height="40%">

Also, add this to your `.bash_profile` and `source` it:

```
export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0
export LIBGL_ALWAYS_INDIRECT=1
```

Source: https://stackoverflow.com/questions/61110603/how-to-set-up-working-x11-forwarding-on-wsl2

**Verify that VcXsrv is running correctly by testing Xcalc:
**
<img src="https://user-images.githubusercontent.com/1320083/144791512-15fa20b7-8dff-4e3f-ba2f-55ade96f3276.png" width="40%" height="40%">

Also, make this change in the code whereever `torch.load` is called:

https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device/62327502#62327502

**Run the real thing:**

`python3 -u environment.py`

<img src="https://user-images.githubusercontent.com/1320083/144791554-3731ce3c-99e2-4877-a766-cbf4664984db.png" width="40%" height="40%">

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
    - All ghosts make an action choice governed by a coin-flip: 10% of the time, it will choose randomly, and the other 90%, it will take a step that brings it closer to Pacman. You can use / adapt the Pathfinder class I gave in the BlindBot package to help with pathfinding or just do something basic like looking at the Manhattan distance between a ghost and Pacman and then choosing the action that minimizes it.
- **MazeUI**: Given a Tk "root window", draw a Tk Canvas on it and all the blocks, characters, and items that make up a maze
