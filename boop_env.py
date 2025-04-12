import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

def env(render_mode=None):
    env = BoopEnv(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class BoopEnv(AECEnv):
    metadata = {
        "name": "boop_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.board_size = 6
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        
        # Action space: (x, y) position on 6x6 board
        self.action_spaces = {
            agent: spaces.Discrete(self.board_size * self.board_size)
            for agent in self.agents
        }
        
        # Observation space: 6x6x2 board (2 channels for each player's pieces)
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=1,
                shape=(self.board_size, self.board_size, 2),
                dtype=np.int8
            )
            for agent in self.agents
        }
        
        self.render_mode = render_mode
        self.board = np.zeros((self.board_size, self.board_size, 2), dtype=np.int8)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.board = np.zeros((self.board_size, self.board_size, 2), dtype=np.int8)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        return self.observe(self.agent_selection), self.infos

    def _boop_pieces(self, x, y, player_idx):
        # Check all 8 surrounding positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.board_size and 0 <= new_y < self.board_size:
                    # If there's a piece in this position
                    if np.any(self.board[new_x, new_y] == 1):
                        # Calculate the position to push to
                        push_x, push_y = new_x + dx, new_y + dy
                        
                        # If the push position is out of bounds, remove the piece
                        if not (0 <= push_x < self.board_size and 0 <= push_y < self.board_size):
                            self.board[new_x, new_y] = 0
                            continue
                            
                        # If the push position is empty, move the piece
                        if np.all(self.board[push_x, push_y] == 0):
                            self.board[push_x, push_y] = self.board[new_x, new_y]
                            self.board[new_x, new_y] = 0

                        # If the push position is occupied, do nothing

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        player_idx = self.agents.index(agent)
        
        # Convert action to coordinates
        x = action // self.board_size
        y = action % self.board_size
        
        # Check if move is valid
        if self.board[x, y, 0] == 1 or self.board[x, y, 1] == 1:
            self.rewards[agent] = -1
            self.terminations[agent] = True
            return self.observe(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        
        # Place piece
        self.board[x, y, player_idx] = 1
        
        # Apply booping mechanic
        self._boop_pieces(x, y, player_idx)
        
        # Check for win condition (3 in a row)
        if self._check_win(x, y, player_idx):
            self.rewards[agent] = 1
            self.rewards[self.agents[1 - player_idx]] = -1
            self.terminations = {agent: True for agent in self.agents}
            return self.observe(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        
        # Switch to next agent
        self.agent_selection = self._agent_selector.next()
        
        self._accumulate_rewards()

        return self.observe(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def observe(self, agent):
        return self.board.copy()

    def _check_win(self, x, y, player_idx):
        # Check horizontal
        for i in range(max(0, x-2), min(self.board_size-2, x+1)):
            if np.all(self.board[i:i+3, y, player_idx] == 1):
                return True
        
        # Check vertical
        for i in range(max(0, y-2), min(self.board_size-2, y+1)):
            if np.all(self.board[x, i:i+3, player_idx] == 1):
                return True
        
        # Check diagonal
        for i in range(-2, 1):
            if 0 <= x+i < self.board_size-2 and 0 <= y+i < self.board_size-2:
                if np.all([self.board[x+i+j, y+i+j, player_idx] == 1 for j in range(3)]):
                    return True
        
        return False

    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            print("\n" + "-" * (self.board_size * 2 + 1))
            for i in range(self.board_size):
                row = "|"
                for j in range(self.board_size):
                    if self.board[i, j, 0] == 1:
                        row += "X|"
                    elif self.board[i, j, 1] == 1:
                        row += "O|"
                    else:
                        row += " |"
                print(row)
                print("-" * (self.board_size * 2 + 1))
            print("\n") 