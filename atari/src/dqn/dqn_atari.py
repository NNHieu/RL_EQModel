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
from dqn.networks import BaselineQNetwork, QMon, QRecurNetwork
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# gym.logger.set_level(gym.logger.DEBUG)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--qnet-name", type=str, default="baseline",
        help="the name of the Q-Network")
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
    out_features = env.single_action_space.nnv
    if qnet_args.name == "baseline":
        return partial(BaselineQNetwork, out_features)
    elif qnet_args.name == "mondeq":
        return partial(QMon, out_features, qnet_args.m, qnet_args.alpha, qnet_args.tol, qnet_args.max_iters)
    elif qnet_args.name == "recur":
        return partial(QRecurNetwork, out_features, qnet_args.num_iters)
    else:
        raise ValueError


def init_wanda(config):
    wandb_cfg = config.logger.wandb
    import wandb

    wandb.init(
        project=wandb_cfg.project_name,
        entity=wandb_cfg.entity,
        sync_tensorboard=True,
        config=vars(config),
        name=config.name,
        monitor_gym=True,
        save_code=True,
    )


def seed_all(seed, is_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = is_deterministic


def main(config):
    # args = parse_args()
    # config.name = f"{args.env_id}__{args.exp_name}_{args.qnet_name}__{args.seed}__{int(time.time())}"
    if config.logger.get("wandb"):
        init_wanda(config)

    writer = SummaryWriter(f"runs/{config.name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    seed_all(config.seed, config.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(config.env, 0, 0, config.capture_video, config.name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    QNetwork = get_qnetwork(config.algo.qnet, envs)
    q_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=config.opt.lr)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        config.algo.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=device,
        optimize_memory_usage=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(config.opt.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            config.opt.start_e, config.opt.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
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
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                if args.qnet_name == "mon":
                    monsp_stats = q_network.core.mon.stats
                    writer.add_scalar("mon_stats/fwd_iters_avg", monsp_stats.fwd_iters.avg, global_step)
                    writer.add_scalar("mon_stats/fwd_err_avg", monsp_stats.fwd_err.avg, global_step)
                    writer.add_scalar("mon_stats/fwd_time_avg", monsp_stats.fwd_time.avg, global_step)
                    writer.add_scalar("mon_stats/bkwd_iters_avg", monsp_stats.bkwd_iters.avg, global_step)
                    writer.add_scalar("mon_stats/bkwd_err_avg", monsp_stats.bkwd_err.avg, global_step)
                    writer.add_scalar("mon_stats/bkwd_time_avg", monsp_stats.bkwd_time.avg, global_step)
                    monsp_stats.reset()
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()