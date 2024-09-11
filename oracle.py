from datetime import datetime

import numpy as np
from str2bool import str2bool as strtobool
from WorldEnvOHEOracle import WorldEnv
from DiscretePySwarms import IntOptimizerPSO
from torch.utils.tensorboard import SummaryWriter
import argparse
np.set_printoptions(threshold=np.inf)

def parse_args():
    parser = argparse.ArgumentParser(description="seed")
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--seed", type=int, help="An example seed", default=0)
    parser.add_argument("--n_particles", type=int, help="The number of particles.", default=10)
    parser.add_argument("--iterations", type=int, help="The number of iterations to run.", default=1000)
    parser.add_argument("--episode_length", type=int, help="The length of an episode.", default=100)
    parser.add_argument("--n_drones", type=int, help="The number of drones in sim.", default=3)
    return parser.parse_args()

def get_fitness(env,actions):
    actions = convert_to_dicts(actions)
    sums = {agent: 0 for agent in actions[0].keys()}#{'drone_0': 0, 'drone_1': 0, 'drone_2': 0}
    terminated = {agent: False for agent in actions[0].keys()}
    observations, infos = env.reset()
    for action_set in actions:
        observations, rewards, terminations, truncations, infos = env.step(action_set)
        for agent, value in rewards.items():
            if terminated[agent] == False:
                sums[agent] += value
            else:
                sums[agent] += 5 #5 reward each timestep for finishing
            if terminations[agent] == True:
                terminated[agent] = True
    total_fitness = sum([v for v in sums.values()])
    return total_fitness
 
def convert_to_dicts(values, group_size=3):
    # Create a list of dictionaries
    dict_list = []
    
    # Ensure the values length is a multiple of group_size
    assert len(values) % group_size == 0, "The list length must be divisible by the group size."

    # Iterate through the values in chunks of group_size
    for i in range(0, len(values), group_size):
        # Create a dictionary for each group
        dict_entry = {f'drone_{j}': values[i + j] for j in range(group_size)}
        dict_list.append(dict_entry)
    
    return dict_list

def generate_actions(environment,timesteps):
    actions_over_timesteps = []
    for i in range(timesteps):
        actions = {agent: environment.action_space(agent).sample() for agent in environment.agents}#[random.randint(0,3) for i in range(num_agents)]
        actions_over_timesteps.append(actions)
    return actions_over_timesteps

def objective_function(solution):
    env = WorldEnv(n_drones=args.n_drones,seed=r_seed, max_timesteps=args.episode_length)
    observations, infos = env.reset()
    score = [-get_fitness(env,sol) for sol in solution]
    return score


if __name__ == "__main__":
    #random seed
    args = parse_args()

    if args.track:
        import wandb
        run = wandb.init(
            project="PSO_RoboticsGridworld",
            entity=None,
            sync_tensorboard=True,
            config={},
            name="ppo_marl_gridworld",
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/ppo_test" + str(datetime.now()))

    max_bound = [3 for _ in range((args.episode_length + 1) * args.n_drones)]
    min_bound = [0 for _ in range((args.episode_length + 1) * args.n_drones)]
    bounds = (min_bound, max_bound)
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

    optimizer = IntOptimizerPSO(n_particles=args.n_particles, dimensions=(args.episode_length*args.n_drones)+3, options=options, bounds=bounds, wandb_track=writer)
    r_seed=args.seed
    cost, pos = optimizer.optimize(objective_function, iters=args.iterations)
    # writer.add_scalar("opt/final_cost", v_loss.item(), global_step)
    wandb.run.summary["cost"] = cost