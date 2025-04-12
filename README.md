# PettingZoo Boop Environment

A PettingZoo environment implementation of the board game Boop. Boop is a 2-player abstract strategy game where players take turns placing pieces on a 6x6 grid, trying to get 3 in a row.

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from boop_env import env

# Create the environment
env = env(render_mode="human")

# Reset the environment
observations, infos = env.reset()

# Play a game
while not env.terminations[env.agent_selection]:
    # Get the current player's observation
    observation = env.observe(env.agent_selection)
    
    # Choose an action (0-35 for the 6x6 grid)
    action = env.action_space(env.agent_selection).sample()
    
    # Step the environment
    env.step(action)
    
    # Render the current state
    env.render()
```

## Game Rules

- The game is played on a 6x6 grid
- Players take turns placing their pieces
- The goal is to get 3 of your pieces in a row (horizontally, vertically, or diagonally)
- Illegal moves (placing a piece on an occupied space) result in a loss
- The game ends when a player wins or makes an illegal move

## Environment Details

- Action Space: Discrete(36) - represents positions on the 6x6 grid
- Observation Space: Box(6, 6, 2) - represents the board state with two channels for each player's pieces
- Rewards: +1 for winning, -1 for losing or making an illegal move
