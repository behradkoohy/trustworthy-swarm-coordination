import functools
import random

import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv


class WorldEnv(ParallelEnv):
    def __init__(self, n_drones=3, render_mode=None, max_x=10, max_y=10, max_timesteps=300):
        self.n_drones = n_drones

        self.possible_agents = ['drone_' + str(x) for x in range(n_drones)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(n_drones)))
        )
        self.render_mode = render_mode
        self.max_x = max_x
        self.max_y = max_y
        self.max_timesteps = max_timesteps

        self.reward_dictionary = {3:-20,4:-5,5:100,0:0}





    def render(self):
        """In future, generate a path diagram to show what the agent has done"""
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=0, high=10, shape=(self.max_x, self.max_y))

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Action space is 4: up, down, left, right
        """
        return Discrete(4)

    def get_observe(self, agent):
        """
        Pass in an agent_id
        """
        agent_grid_value = self.agent_grid_mapping[agent]
        other_agents = [x for x in self.agent_grid_mapping.values() if x != agent_grid_value]
        agent_observe = np.array(self.world_grid)
        for other_agent in other_agents:
            other_agent_positions = np.argwhere(agent_observe == other_agent)
            for other_x, other_y in other_agent_positions:
                agent_observe[other_x, other_y] = 2
        current_agent_positions = np.argwhere(agent_observe == agent_grid_value)
        for curr_x, curr_y in current_agent_positions:
            agent_observe[curr_x, curr_y] = 1
        return np.array(agent_observe, dtype=np.float32)

    def add_safe_zones(self, grid):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        human_positions = np.argwhere(grid == 3)
        for human_x, human_y in human_positions:
            for dx, dy in directions:
                adj_x, adj_y = human_x + dx, human_y + dy
                if 0 <= adj_x < self.max_x and 0 <= adj_y < self.max_y and grid[adj_x, adj_y] == 0:
                    grid[adj_x, adj_y] = 4  # 4 denotes a safe zone
        return grid

    def get_agent_id(self, agent):
        return self.agent_name_mapping[agent]

    def reset(self):
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        """
        In the world np array:
        0 = Empty
        7,8,9 = Drone
        1 = Current Drone
        2 = Other Drone 
        3 = Human
        4 = Human 'Safe Zone'
        5 = Goal
        6 = Obstacle
        """

        self.world_grid = np.zeros((self.max_x, self.max_y))
        starting_locations = []
        while len(starting_locations) < self.n_drones:
            rand_x = random.randint(0, 3)
            rand_y = random.randint(0, 3)
            # rand_x = random.randint(0, 9)
            # rand_y = random.randint(0, 9)
            if (rand_x, rand_y) not in starting_locations:
                starting_locations.append((rand_x, rand_y))

        self.agent_grid_mapping = {agent_id: i for agent_id, i in zip(self.agent_name_mapping.keys(), [7, 8, 9])}
        for (start_x, start_y), drone in zip(starting_locations, [7,8,9]):
            self.world_grid[start_x, start_y] = drone

        # (2, 8), (4, 6)
        # Humans
        self.world_grid[2,6] = 3
        self.world_grid[6,4] = 3

        # # Safe Zones

        self.world_grid[self.max_x-1, self.max_y-1] = 5
        self.world_grid = self.add_safe_zones(self.world_grid)


        observations = {agent: self.get_observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # Update the grid given an action from an agent
    def update_grid_with_action(self, world_grid, agent_num, action):
        reward = 0
        # Find the current position of the agent
        agent_position = np.argwhere(world_grid == agent_num)
        if len(agent_position) == 0:
            return world_grid, 1
        # Extract agent's current coordinates
        current_x, current_y = agent_position[0]
        agent_heuristic = (18 - (abs(9 - current_x) + abs(9 - current_y)))
        # Determine the new position based on the action
        if action == 0:  # Move up
            new_x, new_y = current_x - 1, current_y
        elif action == 1:  # Move right
            new_x, new_y = current_x, current_y + 1
        elif action == 2:  # Move down
            new_x, new_y = current_x + 1, current_y
        elif action == 3:  # Move left
            new_x, new_y = current_x, current_y - 1
        else:
            return world_grid, reward
        # Check if the new position is within the bounds of the grid
        if 0 <= new_x < world_grid.shape[0] and 0 <= new_y < world_grid.shape[1]:
            # Check if the new position is empty (contains a 0)
            if world_grid[new_x, new_y] in [0, 3, 4]:
                # Get reward from zone
                reward = self.reward_dictionary[world_grid[new_x, new_y]] + agent_heuristic
                # Move the agent to the new position
                world_grid[new_x, new_y] = agent_num
                # Set the agent's old position to 0 (empty)
                world_grid[current_x, current_y] = 0
            elif world_grid[new_x, new_y] in [5]:
                reward = self.reward_dictionary[world_grid[new_x, new_y]] + agent_heuristic
                world_grid[new_x, new_y] = 5
                world_grid[current_x, current_y] = 0
            # self.world_grid[2, 6] = 3
            # self.world_grid[6, 4] = 3
            for human_x, human_y in [(2,6), (6,4)]:
                if world_grid[human_x, human_y] == 0:
                    world_grid[human_x, human_y] = 3
            return world_grid, reward
        else:
            return world_grid, reward
        return world_grid, reward

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {}
        terminations = {}
        for agent_name, action in actions.items():
            agent_grid_num = self.agent_grid_mapping[agent_name]
            new_world, reward = self.update_grid_with_action(self.world_grid, agent_grid_num, action)
            # print(new_world)
            self.world_grid = new_world
            rewards[agent_name] = reward
            if rewards[agent_name] == self.reward_dictionary[5]:
                terminations[agent_name] = True
            else:
                terminations[agent_name] = False

        self.num_moves += 1

        env_truncation = self.num_moves >= self.max_timesteps
        truncations = {agent: env_truncation for agent in actions.keys()}

        observations = {agent: self.get_observe(agent) for agent in actions.keys()}

        return observations, rewards, terminations, truncations, {}







if __name__ == '__main__':
    env = WorldEnv(n_drones=3)
    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        print(env.world_grid)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        action_map = {0:'up', 1:'right', 2:'down', 3:'left'}
        print({agent: action_map[int(action)] for agent, action in actions.items()})
        # breakpoint()
        observations, rewards, terminations, truncations, infos = env.step(actions)