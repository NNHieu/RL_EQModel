import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from dqn.networks import BaselineQNetwork, QMon, QRecurNetwork
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer

# gym.logger.set_level(gym.logger.DEBUG)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DEQRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--qnet-name", 
                        choices=['baseline', 'recur', 'mondeq'], 
                        default="baseline",
                        help="The name of the Q-Network. Available value: [baseline, recur, mondeq]")
    
    parser.add_argument("--mon-solver-alpha", type=float,
                        default=1.0,
                        help="monDEQ solver parameter - alpha")
    parser.add_argument("--mon-solver-tol", type=float,
                        default=1e-4,
                        help="monDEQ solver parameter - tolerant")
    parser.add_argument("--mon-solver-max-iters", type=int,
                        default=50,
                        help="monDEQ solver parameter - Max iterations")
    parser.add_argument("--mon-model-m", type=float,
                        default=0.1,
                        help="monDEQ parameter - m")
    
    parser.add_argument("--recur-model-num-iters", type=int,
                        default=10,
                        help="recur model parameter - Number of iters")

    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=100000, #1000000
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if "render.modes" in env.metadata:
            env.metadata["render_modes"] = env.metadata["render.modes"]
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def get_qnetwork(qnet_args, env):
    out_features = env.single_action_space.n
    if qnet_args.qnet_name == "baseline":
        return partial(BaselineQNetwork, out_features)
    elif qnet_args.qnet_name == "mondeq":
        m = qnet_args.mon_model_m
        alpha = qnet_args.mon_solver_alpha
        tol = qnet_args.mon_solver_tol
        max_iters = qnet_args.mon_solver_max_iters

        return partial(QMon, out_features, m, alpha, tol, max_iters)
    elif qnet_args.qnet_name == "recur":
        return partial(QRecurNetwork, out_features, qnet_args.recur_model_num_iters)
    else:
        raise ValueError


def seed_all(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def bo_func(args, learning_rate, mon_model_m, mon_solver_tol):
    run_name = f"{args.env_id}__{args.exp_name}_{args.qnet_name}__{args.seed}__{int(time.time())}"
    if learning_rate is not None:
        args.learning_rate = learning_rate
    if mon_model_m is not None:
        args.mon_model_m = mon_model_m
    if mon_solver_tol is not None:
        args.mon_solver_tol = mon_solver_tol
    running_reward = None

    seed_all(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    QNetwork = get_qnetwork(args, envs)
    q_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, envs.single_action_space, device=device, optimize_memory_usage=True
    )

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            logits = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(logits, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                # log_episode(writer, global_step, info, q_network)
                running_reward = (
                    info["episode"]["r"] if running_reward is None else running_reward * 0.99 + info["episode"]["r"] * 0.01
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            # if global_step % 100 == 0:
            #     writer.add_scalar("losses/td_loss", loss, global_step)
            #     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            #     # print("SPS:", int(global_step / (time.time() - start_time)))
            #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    envs.close()
    return running_reward


if __name__ == "__main__":
    args = parse_args()
    # Bounded region of parameter space
    pbounds = {"learning_rate": (1e-5, 1e-3)}

    def black_box_function(learning_rate):
        bo_func(args, learning_rate, None, None)

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    logger = JSONLogger(path="./atadqnmon_tuning_logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )
    print(optimizer.max)
