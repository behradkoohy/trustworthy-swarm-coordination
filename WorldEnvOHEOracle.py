import functools
import random

import numpy as np
import numpy.random
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

"""
0 = Humans
1 = Safe Zones
2 = Targets
3.. = Drones
"""


class WorldEnv(ParallelEnv):
    def __init__(self, n_drones=3, n_humans=2, drone_locations = None, human_locations = [(2,6), (6,4)], targets = [(9,9)], reward_human=-10000, reward_safe_zone=-5000, reward_target=100, render_mode=None, max_x=10, max_y=10, max_timesteps=300,seed=0):
        self.n_drones = n_drones
        self.n_humans = n_humans
        self.possible_agents = ['drone_' + str(x) for x in range(n_drones)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(n_drones)))
        )
        self.render_mode = render_mode

        self.max_x = max_x
        self.max_y = max_y
        self.max_timesteps = max_timesteps
        self.humans = human_locations
        self.drone_init_locations = drone_locations
        self.targets = targets
        self.all_grids = np.array([])
        self.unavailable_locs = np.array([])
        self.reward_dictionary = {0: reward_human, 1: reward_safe_zone, 2: reward_target}

        self.seed = seed

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
        Pass in an agent_id and get an observation space for them.
        """
        agent_grid_value = self.agent_grid_mapping[agent]
        other_agents = [x for x in self.agent_grid_mapping.values() if x != agent_grid_value]
        agent_observe = np.array(self.all_grids)

        # Create a OHE grid for other agents and the current agent
        other_agents_grid = np.zeros((self.max_x, self.max_y))
        current_agents_grid = np.zeros((self.max_x, self.max_y))

        # Fill out the other agents grid
        other_agent_positions = []
        for other_agent in other_agents:
            other_agent_positions.append(np.argwhere(agent_observe[other_agent] == 1)[0])
        for other_x, other_y in other_agent_positions:
            other_agents_grid[other_x, other_y] = 1

        # FIll out the current agent grid
        current_agent_position = np.argwhere(agent_observe[agent_grid_value] == 1)[0]
        current_agents_grid[current_agent_position[0], current_agent_position[1]] = 1

        # Concatenate observations to the overall OHE environment
        agent_observe = np.append(agent_observe, np.array([current_agents_grid]), axis=0)
        agent_observe = np.append(agent_observe, np.array([other_agents_grid]), axis=0)
        return np.array(agent_observe, dtype=np.float32)

    """
    When agent start points are not provided, this function initialises their position randomly.
    """

    def get_agent_start_point_OHE(self, world_grid, unavailable_locs):
        loc = None
        random.seed(self.seed)
        while world_grid.sum() == 0:
            rand_x = random.randint(0, int(self.max_x/2))  # Make it self.max_x
            rand_y = random.randint(0, int(self.max_y/2))
            loc = (rand_x, rand_y)
            if loc not in unavailable_locs:
                world_grid[rand_x, rand_y] = 1
                unavailable_locs.append(loc)
        return world_grid, unavailable_locs

    """
    OHE world grid with locs
    """

    def get_start_point_OHE_from_locations(self, world_grid, locations):
        for loc_x, loc_y in locations:
            world_grid[loc_x, loc_y] = 1
        return world_grid

    """
    Generate safe zones around all the human positions. This uses the human grid and fills out around the humans. 
    """

    def add_safe_zones(self, human_grid, world_grid):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        human_positions = np.argwhere(human_grid == 1)
        for human_x, human_y in human_positions:
            for dx, dy in directions:
                adj_x, adj_y = human_x + dx, human_y + dy
                if 0 <= adj_x < self.max_x and 0 <= adj_y < self.max_y and world_grid[adj_x, adj_y] == 0:
                    world_grid[adj_x, adj_y] = 1
        return world_grid

    """
    Initialise the starting grid
    """

    def initialise_grid(self, humans, targets):

        world_grid = np.zeros((self.max_x, self.max_x))
        all_grids = []
        unavailable_locs = []

        # ADD THE GRIDS FOR HUMANS AND SAFE ZONES
        human_grid = self.get_start_point_OHE_from_locations(np.array(world_grid), humans)
        safe_grid = self.add_safe_zones(human_grid, np.array(world_grid))
        all_grids.append(human_grid)
        all_grids.append(safe_grid)

        # ADD THE TARGET OHE GRID
        all_grids.append(self.get_start_point_OHE_from_locations(np.array(world_grid), targets))

        # IF THE DRONE LOCATIONS ARE PROVIDED, ADD THESE AS OHE GRIDS.
        if self.drone_init_locations != None:
            for loc_x, loc_y in self.drone_init_locations:
                temp_world_grid = np.array(world_grid)
                temp_world_grid[loc_x, loc_y] = 1
                all_grids.append(temp_world_grid)
                unavailable_locs.append((loc_x, loc_y))
        else:  # IF THE DRONE LOCATIONS ARE NOT PROVIDED, RANDOMLY INITIALISE THEM AND ADD THESE AS OHE GRIDS.
            for i in range(self.n_drones):
                new_agent_grid, unavailable_locs = self.get_agent_start_point_OHE(np.array(world_grid),
                                                                                  unavailable_locs)
                all_grids.append(new_agent_grid)

        return all_grids, unavailable_locs

    def get_agent_id(self, agent):
        return self.agent_name_mapping[agent]

    def reset(self):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        all_grids, unavailable_locs = self.initialise_grid(self.humans, self.targets)
        self.all_grids = all_grids
        self.unavailable_locs = unavailable_locs
        self.agent_grid_mapping = {agent_id: i for agent_id, i in zip(self.agent_name_mapping.keys(), range(3,
            3 + self.n_drones))}  # DRONES START FROM 3 ONWARDS
        observations = {agent: self.get_observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def get_action_masks(self, all_grids, agent_num):
        agents_grid = np.array(all_grids[agent_num+3])
        agent_position = np.argwhere(agents_grid == 1)
        # breakpoint()
        current_x, current_y = agent_position[0]
        act_mask = []
        for act in [0, 1, 2, 3]:
            if act == 0:  # Move up
                new_x, new_y = current_x - 1, current_y
            elif act == 1:  # Move right
                new_x, new_y = current_x, current_y + 1
            elif act == 2:  # Move down
                new_x, new_y = current_x + 1, current_y
            elif act == 3:  # Move left
                new_x, new_y = current_x, current_y - 1
            if not (0 <= new_x < self.max_x and 0 <= new_y < self.max_y): # if the action takes the drone out of bounds
                act_mask.append(False)
            else:
                if any(all_grids[agt_world][new_x, new_y] == 1 for agt_world in range(3, 3+self.n_drones)):
                    act_mask.append(False)
                else:
                    act_mask.append(True)
        return act_mask


    # UPDATE THE CORRESPONDING OHE GRID GIVEN AN ACTION FROM AN AGENT
    def update_grid_with_action(self, all_grids, agent_num, action):

        # Get the grid relating to the specified agent
        agents_grid = all_grids[agent_num]
        reward = 0

        # Find the current position of the agent
        agent_position = np.argwhere(agents_grid == 1)
        current_x, current_y = agent_position[0]
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
            return all_grids, reward

        # Check if the new position is within the bounds of the grid
        if 0 <= new_x < self.max_x and 0 <= new_y < self.max_y:
            for key, value in self.reward_dictionary.items():  # Loops through the dictionary of potential rewards and check if any item is in the new zone. Return that reward if so.
                if all_grids[key][new_x, new_y] == 1:
                    reward += value
            agents_grid[new_x, new_y] = 1
            agents_grid[current_x, current_y] = 0
            return all_grids, reward
        else:
            return all_grids, reward

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {}
        terminations = {}

        # PERFORM ACTIONS FOR EACH AGENT
        new_agent_locations = []
        for agent_name, action in actions.items():
            agent_grid_num = self.agent_grid_mapping[agent_name]
            new_world, reward = self.update_grid_with_action(np.array(self.all_grids), agent_grid_num, action)
            new_agent_location = np.argwhere(new_world[agent_grid_num] == 1)[0]
            already_exists = any(np.array_equal(new_agent_location, arr) for arr in new_agent_locations)

            # IF THE NEW AGENT LOCATION IS NOT ALREADY OCCUPIED, MOVE THE AGENT TO THE NEW ZONE. OTHERWISE, DON'T MOVE
            if not already_exists:
                self.all_grids = new_world
                rewards[agent_name] = reward
                new_agent_locations.append(new_agent_location)
            else:
                rewards[agent_name] = 0  # PENALTY FOR TRYING TO GO TO SAME SPOT AS ANOTHER AGENT?

            # Termination condition.
            target_location = np.argwhere(new_world[2] == 1)[0]
            if np.array_equal(target_location, new_agent_location):
                terminations[agent_name] = True
            else:
                terminations[agent_name] = False

        self.num_moves += 1

        env_truncation = self.num_moves >= self.max_timesteps
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {agent: self.get_observe(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, {}


if __name__ == '__main__':
    env = WorldEnv(n_drones=3)
    observations, infos = env.reset()
    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        action_map = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
        observations, rewards, terminations, truncations, infos = env.step(actions)