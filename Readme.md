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

1. monDEQ network
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

2. Recurrent model
- `--recur-model-num-iters`