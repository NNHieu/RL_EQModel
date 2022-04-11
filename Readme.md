# DEQ RL model
Based on [CleanRL repo](https://github.com/vwxyzjn/cleanrl).

## Get started

1. Set up environment
    ```bash
    conda env create -p .venv -f environment.yml
    conda activate ./.venv
    ```
2. Basic usage
    ```Shell
    python atari/dqn_atari.py --env-id BreakoutNoFrameskip-v4 --seed 1 --qnet-name baseline --total-timesteps 1000000

    # open another temrminal and enter
    tensorboard --logdir runs
    ```

## Tuning
1. Model hyper-parameter
    
    a. monDEQ network
    - `--mon-model-m`: (>0)      control how strongly monotone the operator F is
    - `--mon-solver-tol`:        monDEQ solver parameter - tolerent
    - `--mon-solver-max-iters`:  monDEQ solver parameter - Max iterations
    - `--mon-solver-alpha`: (>0) monDEQ solver parameter - alpha ( Peaceman-Rachford solver will converge for any choice of `alpha` for strongly monotone F , though the convergence speed will often vary substantially based upon α.)

    Example: 
    ```bash
    python atari/dqn_atari.py --env-id BreakoutNoFrameskip-v4 \
    --qnet-name mondeq --mon-model-m 0.1 \
    --mon-solver-tol 1e-4 --mon-solver-max-iters 30 --mon-solver-alpha 0.5 \
    --seed 1 --total-timesteps 10000000
    ```

    b. Recurrent model
    - `--recur-model-num-iters`

2. Training hyper-parameter

    We use Adam optimizer 
    - `--learning-rate`

## Get Documentation
You can directly obtained the documentation by using the --help flag.

```bash
python atari/dqn_atari.py --help
usage: dqn_atari.py [-h] [--exp-name EXP_NAME] [--seed SEED] [--torch-deterministic [TORCH_DETERMINISTIC]] [--cuda [CUDA]] [--track [TRACK]] [--wandb-project-name WANDB_PROJECT_NAME]
                    [--wandb-entity WANDB_ENTITY] [--capture-video [CAPTURE_VIDEO]] [--qnet-name {baseline,recur,mondeq}] [--mon-solver-alpha MON_SOLVER_ALPHA]
                    [--mon-solver-tol MON_SOLVER_TOL] [--mon-solver-max-iters MON_SOLVER_MAX_ITERS] [--mon-model-m MON_MODEL_M] [--recur-model-num-iters RECUR_MODEL_NUM_ITERS]
                    [--env-id ENV_ID] [--total-timesteps TOTAL_TIMESTEPS] [--learning-rate LEARNING_RATE] [--buffer-size BUFFER_SIZE] [--gamma GAMMA]
                    [--target-network-frequency TARGET_NETWORK_FREQUENCY] [--max-grad-norm MAX_GRAD_NORM] [--batch-size BATCH_SIZE] [--start-e START_E] [--end-e END_E]
                    [--exploration-fraction EXPLORATION_FRACTION] [--learning-starts LEARNING_STARTS] [--train-frequency TRAIN_FREQUENCY]

optional arguments:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   the name of this experiment
  --seed SEED           seed of the experiment
  --torch-deterministic [TORCH_DETERMINISTIC]
                        if toggled, `torch.backends.cudnn.deterministic=False`
  --cuda [CUDA]         if toggled, cuda will be enabled by default
  --track [TRACK]       if toggled, this experiment will be tracked with Weights and Biases
  --wandb-project-name WANDB_PROJECT_NAME
                        the wandb's project name
  --wandb-entity WANDB_ENTITY
                        the entity (team) of wandb's project
  --capture-video [CAPTURE_VIDEO]
                        weather to capture videos of the agent performances (check out `videos` folder)
  --qnet-name {baseline,recur,mondeq}
                        The name of the Q-Network. Available value: [baseline, recur, mondeq]
  --mon-solver-alpha MON_SOLVER_ALPHA
                        monDEQ solver parameter - alpha
  --mon-solver-tol MON_SOLVER_TOL
                        monDEQ solver parameter - tolerent
  --mon-solver-max-iters MON_SOLVER_MAX_ITERS
                        monDEQ solver parameter - Max iterations
  --mon-model-m MON_MODEL_M
                        monDEQ parameter - m
  --recur-model-num-iters RECUR_MODEL_NUM_ITERS
                        recur model parameter - Number of iters
  --env-id ENV_ID       the id of the environment
  --total-timesteps TOTAL_TIMESTEPS
                        total timesteps of the experiments
  --learning-rate LEARNING_RATE
                        the learning rate of the optimizer
  --buffer-size BUFFER_SIZE
                        the replay memory buffer size
  --gamma GAMMA         the discount factor gamma
  --target-network-frequency TARGET_NETWORK_FREQUENCY
                        the timesteps it takes to update the target network
  --max-grad-norm MAX_GRAD_NORM
                        the maximum norm for the gradient clipping
  --batch-size BATCH_SIZE
                        the batch size of sample from the reply memory
  --start-e START_E     the starting epsilon for exploration
  --end-e END_E         the ending epsilon for exploration
  --exploration-fraction EXPLORATION_FRACTION
                        the fraction of `total-timesteps` it takes from start-e to go end-e
  --learning-starts LEARNING_STARTS
                        timestep to start learning
  --train-frequency TRAIN_FREQUENCY
                        the frequency of training
```