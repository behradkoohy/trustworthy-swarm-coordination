import datetime
import argparse
import math
import random
import warnings

from str2bool import str2bool as strtobool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from welford import Welford
from WelfordR import WelfordR
from WorldEnvOHE import WorldEnv


def parse_args():
    parser = argparse.ArgumentParser()
    """
    Experiment parameters
    """
    parser.add_argument("--exp_name", type=str, default="CRL_PPO")


    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )

    """
    Environment parameters
    """
    parser.add_argument(
        "--episode_length",
        type=int,
        default=500,
        nargs="?",
        const=True,
        help="The maximum length in steps of an episode before the episode is truncated",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2500,
        help="total episodes of the experiments",
    )

    """
    Default PPO Parameters
    """
    parser.add_argument(
        "--ent-coef", type=float, default=0.05, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.1, help="coefficient of the value function"
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.1,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.9999, help="the discount factor gamma"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128 * 4, help="the discount factor gamma"
    )
    # parser.add_argument("--update-epochs", type=int, default=0.99999,
    #                     help="the discount factor gamma")

    """
    Our PPO Parameters
    """
    parser.add_argument(
        "--agent-reward-norm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, PPO will compute the mean and std for the reward and use it to normalise the reward",
    )
    parser.add_argument(
        "--warmup-arn",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, PPO will pre-initialise the agent reward norm by running an environment and setting mean/var for the reward norm to this. If false, ARN will initialise with mean and var of 0.",
    )
    # parser.add_argument("--global-reward-norm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #                     help="if toggled, PPO will compute the mean and std for the reward and use it to normalise the reward")
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle whether Generalised Advantage Estimation is used to calculate returns.",
    )
    parser.add_argument(
        "--gae-coef",
        type=float,
        default=0.9999,
        help="The GAE Coefficient used for return estimation.",
    )
    parser.add_argument(
        "--remove-dead-agents",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle whether to remove dead agent (s,a,r,s') after they reach a terminal state.",
    )
    parser.add_argument(
        "--distinct-actor-critic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If true, actor and critic will be two separate NNs. If false, actor and critic will be one NN with separate output heads.",
    )
    parser.add_argument(
        "--reward-clip",
        type=int,
        default=0,
        help="Defines whether the rewards from the agent are clipped. If set to 0, rewards are not clipped.",
    )
    parser.add_argument(
        "--action-masks",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="If true, invalid agent actions will be masked. If false, they will have no effect.",
    )

    args = parser.parse_args()
    return args


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        if args.distinct_actor_critic:
            self.actor = nn.Sequential(
                self._layer_init(nn.Conv2d(5, 16, 4, padding=1)),
                nn.ReLU(),
                self._layer_init(nn.Conv2d(16, 16, 3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
                self._layer_init(nn.Linear(1296, 128)),
                nn.ReLU(),
                self._layer_init(nn.Linear(128, num_actions), std=0.01),
            )
            self.critic = nn.Sequential(
                self._layer_init(nn.Conv2d(5, 16, 4, padding=1)),
                nn.ReLU(),
                self._layer_init(nn.Conv2d(16, 16, 3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
                self._layer_init(nn.Linear(1296, 128)),
                nn.ReLU(),
                self._layer_init(nn.Linear(128, 1)),
            )
        else:
            self.network = nn.Sequential(
                self._layer_init(nn.Conv2d(5, 16, 4, padding=1)),
                nn.ReLU(),
                self._layer_init(nn.Conv2d(16, 16, 3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
                self._layer_init(nn.Linear(1296, 128)),
                nn.ReLU(),
            )
            self.actor = self._layer_init(nn.Linear(128, num_actions), std=0.01)
            self.critic = self._layer_init(nn.Linear(128, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        if args.distinct_actor_critic:
            return self.critic(x)
        else:
            return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, action_masks=None):
        if args.distinct_actor_critic:
            logits = self.actor(x)
        else:
            hidden = self.network(x)
            logits = self.actor(hidden)
        if not args.action_masks:
            probs = Categorical(logits=logits)
        if args.action_masks:
            if action_masks is None:
                raise Exception(
                    "Action Masks called but none passed to action and value funct."
                )
            probs = CategoricalMasked(logits=logits, masks=action_masks)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            (self.critic(x) if args.distinct_actor_critic else self.critic(hidden)),
        )


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    # breakpoint()

    # print(obs.keys(), type(obs))
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    # obs = np.expand_dims(obs, axis=-1)
    # obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.agents)}
    return x


if __name__ == "__main__":
    args = parse_args()
    if args.track:
        import wandb

        run = wandb.init(
            project="RoboticsGridworld",
            entity=None,
            sync_tensorboard=True,
            config={},
            name="ppo_marl_gridworld",
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/ppo_test" + str(datetime.datetime.now()))
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = args.ent_coef
    vf_coef = args.vf_coef
    clip_coef = args.clip_coef
    gamma = args.gamma
    batch_size = args.batch_size
    frame_size = (10, 10)
    max_cycles = 1000
    total_episodes = args.num_episodes

    """ ENV SETUP """
    env = WorldEnv(max_timesteps=args.episode_length)
    num_agents = 3
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, 5, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)
    rb_action_masks = torch.zeros((max_cycles, num_agents, num_actions)).to(device)
    rb_mask = torch.zeros((max_cycles+1, num_agents)).to(device)
    global_step = 0

    if args.agent_reward_norm:
        agent_welford = {agent: WelfordR() for agent in env.possible_agents}

    if args.warmup_arn:
        if not args.agent_reward_norm:
            warnings.warn(
                "ARN is set to False but Warmup ARN is set to true. Warmup ARN will have no impact. Is this what you meant to do?"
            )
        else:
            obs, infos = env.reset()
            print("Warming up ARN with random actions")
            for x in range(100):
                for step in range(0, args.episode_length):
                    actions = {
                        agent: env.action_space(agent).sample() for agent in env.agents
                    }
                    observations, rewards, terms, truncs, infos = env.step(actions)
                    for agt in env.agents:
                        agent_welford[agt].add(np.array([rewards[agt]]))
                    if (
                            all([terms[a] for a in terms])
                            or all([truncs[a] for a in truncs])
                            or terms == []
                    ):
                        break
            print("Warm up complete.")

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(args.num_episodes):
        if args.anneal_lr:
            frac = 1.0 - (episode / args.num_episodes)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rb_mask[0] = torch.tensor([False, False, False])
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            t_seed = random.randint(0,30)
            # print(t_seed)
            next_obs, info = env.reset(seed=t_seed)
            # reset the episodic return
            total_episodic_return = 0

            dead_agents = []
            alive_agents = [agt for agt in env.agents]
            dead_agent_step = {
                agt: args.episode_length for agt in env.agents
            }
            safe_zones_hit = 0
            # each episode has num_steps
            for step in range(0, args.episode_length):
                # rollover the observation
                global_step += 1
                obs = batchify_obs(next_obs, device)

                # get action masks for agent
                if args.action_masks:
                    # unbatched_masks = [env.get_action_masks(env.all_grids, n) for n, ob in enumerate(obs)]
                    # breakpoint()
                    try:
                        masks = torch.tensor(
                            [env.get_action_masks(env.all_grids, env.agent_name_mapping[agt]) for agt in env.agents]
                        )
                        actions, logprobs, _, values = agent.get_action_and_value(
                            obs, action_masks=masks
                        )
                    except Exception as e:
                        breakpoint()
                        raise e
                else:
                    # get action from the agent
                    actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )


                # normalise the agent reward via scaling and shaping
                if args.agent_reward_norm:
                    for agt in env.possible_agents:
                        agent_welford[agt].add(np.array([rewards[agt]]))
                        if global_step < 2 or agent_welford[agt].var_s[0] == 0.0:
                            pass
                        else:
                            rewards[agt] = (rewards[agt] - agent_welford[agt].mean[0]) / math.sqrt(agent_welford[agt].var_s[0] + 1e-8)

                for agt, reward in rewards.items():
                    if reward <= -1:
                        safe_zones_hit += 1
                    if args.reward_clip > 0:
                        rewards[agt] = np.clip(
                            reward, -args.reward_clip, args.reward_clip
                        )

                # breakpoint()
                # if episode == 99:
                #     env.show_grid(env.all_grids)
                # for agt in dead_agents:
                #     terms[agt] = True
                #     rewards[agt] = 0.0

                # before we remove any newly dead agents, we need to make sure everything else is on point
                if len(env.agents) < 3:
                    # unbatched_next_obs = unbatchify(next_obs, env)
                    unbatched_obs = unbatchify(obs, env)
                    unbatched_actions = unbatchify(actions, env)
                    unbatched_logprobs = unbatchify(logprobs, env)
                    unbatched_values = unbatchify(values[:,0], env)
                    if args.action_masks:
                        unbatched_masks = {agt: m for agt, m in zip(env.agents, masks.tolist())}


                    for agt in dead_agents:
                        # next_obs[agt] = np.zeros((5,10,10), dtype=np.float32)

                        unbatched_obs[agt] = np.zeros((5,10,10))

                        rewards[agt] = 0.0
                        terms[agt] = True
                        unbatched_actions[agt] = 3
                        unbatched_logprobs[agt] = 0
                        unbatched_values[agt] = 0
                        if args.action_masks:
                            unbatched_masks[agt] = [False, False, False, False]

                    obs = batchify(unbatched_obs, device)
                    actions = batchify(unbatched_actions, device)
                    logprobs = batchify(unbatched_logprobs, device)
                    values = batchify(unbatched_values, device)
                    if args.action_masks:
                        masks = batchify(unbatched_masks, device)


                # Deal with a dead agent, remove from game and add to the dead agent repo
                for agt in (agt for agt in terms if terms[agt] and agt not in dead_agents):
                    dead_agent_step[agt] = step
                    env.agents.remove(agt)
                    alive_agents.remove(agt)
                    dead_agents.append(agt)

                # if episode == 99:
                #     env.show_grid(env.all_grids)

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_mask[step+1] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()
                if args.action_masks:
                    rb_action_masks[step] = masks
                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                # print({r:int(x) for r,x in rewards.items()}, rb_rewards[step], total_episodic_return, actions, env.agent_position)

                # if we reach termination or truncation, end
                if (
                        all([terms[a] for a in terms])
                        or all([truncs[a] for a in truncs])
                        or terms == []
                ):
                    end_step = step
                    # print(
                    #     episode,
                    #     end_step,
                    #     terms,
                    #     # truncs,
                    #     total_episodic_return,
                    #     sum(total_episodic_return),
                    #     # dead_agent_step,
                    #     sum(dead_agent_step.values()),
                    #     agent_welford if args.agent_reward_norm else None,
                    # )
                    print("SEED:", t_seed, step, safe_zones_hit, episode)
                    # env.show_grid(env.all_grids)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                if args.gae:
                    rb_advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_coef * rb_terms[t + 1] * lastgaelam
                    )
                else:
                    rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)
        b_action_masks = torch.flatten(
            rb_action_masks[:end_step], start_dim=0, end_dim=1
        )

        if args.remove_dead_agents:
            # for n, agt_end in enumerate(dead_agent_step.values()):
            #     rb_terms[agt_end][n] = 0
            # b_terms = torch.flatten(rb_terms[:end_step])
            # b_terms = ~(b_terms > 0)


            b_mask = torch.logical_not(torch.flatten(rb_mask[:end_step], start_dim=0, end_dim=1))

            b_obs = b_obs[b_mask]
            b_logprobs = torch.masked_select(b_logprobs, b_mask)
            b_actions = torch.masked_select(b_actions, b_mask)
            b_returns = torch.masked_select(b_returns, b_mask)
            b_values = torch.masked_select(b_values, b_mask)
            b_advantages = torch.masked_select(b_advantages, b_mask)

            if args.action_masks:
                b_action_masks = b_action_masks[b_mask]

            # for agt_end in dead_agent_step.values():
            #     b_terms[agt_end] = False
            # b_terms = ~b_terms

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))

        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + (batch_size)
                batch_index = b_index[start:end]
                if not args.action_masks:
                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                else:
                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index],
                        b_actions.long()[batch_index],
                        action_masks=b_action_masks[batch_index],
                    )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantages
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "losses/sum_total_episodic_return",
            np.sum(total_episodic_return),
            global_step,
        )
        writer.add_scalar("eval/episode_length", end_step, global_step)
        writer.add_scalar("eval/drone_0_reward", total_episodic_return[0], global_step)
        writer.add_scalar("eval/drone_1_reward", total_episodic_return[1], global_step)
        writer.add_scalar("eval/drone_2_reward", total_episodic_return[2], global_step)
        writer.add_scalar("eval/t_seed", t_seed, global_step)
        if args.agent_reward_norm:
            writer.add_scalar(
                "eval/drone_0_mean", agent_welford["drone_0"].mean[0], global_step
            )
            writer.add_scalar(
                "eval/drone_1_mean", agent_welford["drone_1"].mean[0], global_step
            )
            writer.add_scalar(
                "eval/drone_2_mean", agent_welford["drone_2"].mean[0], global_step
            )
            writer.add_scalar(
                "eval/drone_0_var", agent_welford["drone_0"].var_s[0], global_step
            )
            writer.add_scalar(
                "eval/drone_1_var", agent_welford["drone_1"].var_s[0], global_step
            )
            writer.add_scalar(
                "eval/drone_2_var", agent_welford["drone_2"].var_s[0], global_step
            )

    torch.save(agent.actor.state_dict(), 'actor.model')
    torch.save(agent.critic.state_dict(), 'critic.model')

        # print(f"Training episode {episode}")
        # print(f"Episodic Return: {np.mean(total_episodic_return)}")
        # print(f"Episode Length: {end_step}")
        # print("")
        # print(f"Value Loss: {v_loss.item()}")
        # print(f"Policy Loss: {pg_loss.item()}")
        # print(f"Old Approx KL: {old_approx_kl.item()}")
        # print(f"Approx KL: {approx_kl.item()}")
        # print(f"Clip Fraction: {np.mean(clip_fracs)}")
        # print(f"Explained Variance: {explained_var.item()}")
        # print("\n-------------------------------------------\n")
    exit()
    agent.eval()
    args.action_masks = False
    episode_length = []
    with torch.no_grad():
        for seed in range(30):
            dead_agents = []
            next_obs, infos = env.reset(seed=seed)
            for step in range(args.episode_length):
                try:
                    obs = batchify_obs(next_obs, device)
                except ValueError as e:
                    breakpoint()
                    raise e
                masks = torch.tensor([env.get_action_masks(env.all_grids, env.agent_name_mapping[agt]) for agt in env.agents])
                actions, logprobs, _, values = agent.get_action_and_value(obs, action_masks=masks)
                next_obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                # env.show_grid(env.all_grids)

                for agt in (agt for agt in terms if terms[agt] and agt not in dead_agents):
                    dead_agents.append(agt)
                    env.agents.remove(agt)

                if (
                        all([terms[a] for a in terms])
                        or all([truncs[a] for a in truncs])
                        or terms == []
                        or env.agents == []
                ):
                    end_step = step

                    break

    """ RENDER THE POLICY """
    # env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
    # env = color_reduction_v0(env)
    # env = resize_v1(env, 64, 64)
    # env = frame_stack_v1(env, stack_size=4)
    #
    # agent.eval()
    #
    # with torch.no_grad():
    #     # render 5 episodes out
    #     for episode in range(5):
    #         obs, infos = env.reset(seed=None)
    #         obs = batchify_obs(obs, device)
    #         terms = [False]
    #         truncs = [False]
    #         while not any(terms) and not any(truncs):
    #             actions, logprobs, _, values = agent.get_action_and_value(obs)
    #             obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
    #             obs = batchify_obs(obs, device)
    #             terms = [terms[a] for a in terms]
    #             truncs = [truncs[a] for a in truncs]