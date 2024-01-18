import os
import time
import random

import numpy as np
import pandas as pd
import torch


def get_env_log_data(env, mean_reward, start_time):
    """Get log data from the environment."""
    # Try-except to handle different env-wrappers
    try:
        episode_info = env.env_method('return_episode_info')[0]
    except AttributeError:
        episode_info = env.return_episode_info()
    discharge_count = len(list(filter(lambda x: (x > 0), episode_info['bes_energy_flows'])))
    charge_count = len(list(filter(lambda x: (x < 0), episode_info['bes_energy_flows'])))
    stats = {
        'reward_sum': mean_reward,
        'degr_cost_sum': sum(episode_info['degr_costs']),
        'compute_time': time.time() - start_time,
        'avg_soc': sum(episode_info['socs']) / len(episode_info['socs']),
        'num_charging': charge_count,
        'num_discharging': discharge_count,
    }
    if episode_info['e_balances']:
        num_oversupply = len(list(filter(lambda x: (x > 0), episode_info['e_balances'])))
        avg_oversupply = sum(list(filter(lambda x: (x > 0), episode_info['e_balances']))) / num_oversupply
        num_undersupply = len(list(filter(lambda x: (x < 0), episode_info['e_balances'])))
        avg_undersupply = sum(list(filter(lambda x: (x > 0), episode_info['e_balances']))) / num_undersupply

        demand_balancing_stats = {
            'num_oversupply': num_oversupply,
            'avg_oversupply': avg_oversupply,
            'num_undersupply': num_undersupply,
            'avg_undersupply': avg_undersupply,
        }
        stats = stats | demand_balancing_stats

    log_data = stats | episode_info

    return log_data


def create_stats_file(path, exp_params):
    """Create statistic file for experiments with multiple random seeds."""
    stats = pd.DataFrame(columns=[i for i in range(exp_params['n_episodes'] + 1)])
    if exp_params['eval_while_training']:
        eval_episodes = int(exp_params['n_episodes'] * 8760 / exp_params['eval_freq'])
        eval_stats = pd.DataFrame(columns=[i for i in range(eval_episodes)])
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('eval_monitor.csv'):
                df = pd.read_csv(os.path.join(subdir, file))
                eval_stats.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
            if file.endswith('train_monitor.csv') and 'eval' not in file:
                df = pd.read_csv(os.path.join(subdir, file))
                stats.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
                count += 1

    mean = stats.mean(axis=0)
    std = stats.std(axis=0)
    stats.loc['mean'] = mean
    stats.loc['std'] = std
    stats.to_csv(os.path.join(path, 'stats.csv'))
    print('Mean episodic rewards over all runs: ')
    print()
    print(mean)

    if exp_params['eval_while_training']:
        mean = eval_stats.mean(axis=0)
        std = eval_stats.std(axis=0)
        eval_stats.loc['mean'] = mean
        eval_stats.loc['std'] = std
        eval_stats.to_csv(os.path.join(path, 'eval_stats.csv'))


def set_seeds(seed):
    """Fixes the random seed for all relevant packages.

    :param seed: Seed to be set
    :type seed: int
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
