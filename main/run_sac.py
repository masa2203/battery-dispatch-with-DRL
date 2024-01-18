import os
import json
from typing import Any

from environments.environment import *
from environments.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_schedule
from utils.utils import create_stats_file
from train.train import train_rl_agent


# CHOOSE ENVIRONMENT/CASE STUDY
# -----------------------------------------------------------------
# AB-EA
ENV = EA_BESS
ENV_KWARGS = al4_bat_ea
ENV_KWARGS['state_vars'] = ['pool_price', 'cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m']

# DE-LF/REU
# ENV = HES_BESS_Env1
# ENV_KWARGS = de1_bat_hes
# ENV_KWARGS['state_vars'] = ['pool_price', 're_power', 'cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m']
# -----------------------------------------------------------------

# LOG
CREATE_LOG = False
VERBOSE = 0
LOGGER_TYPE = ["csv", "tensorboard"]
SAVE_PATH = os.path.join('../log/', ENV_KWARGS['env_name'], 'sac/run', input('Save in folder: ')) \
    if CREATE_LOG else None

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 50,
    'seed': 22,
    # Reward wrapper
    'inactive_penalty': None,
    # 'inactive_penalty': {'patience': 10, 'penalty': 190},
    'action_corr_penalty': None,
    'soc_penalty': None,
    # Env
    'flatten_obs': True,
    'frame_stack': 2,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': 8760 * 1,
}

# SAC PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'learning_rate': 2.5e-4,  # Default: 3e-4
    # 'learning_rate': linear_schedule(2.5e-4),  # Default: 3e-4
    'buffer_size': 500_000,  # Default: 1e6
    'learning_starts': 1000,  # Default: 100
    'batch_size': 512,  # Default: 256
    'tau': 0.006985,  # Default: 0.005
    'gamma': 0.965175,  # Default: 0.99
    'train_freq': 35,  # Default: 1
    'gradient_steps': -1,  # Default: 1
    'action_noise': None,
    'ent_coef': 'auto_0.1',  # Default: 'auto', use float alternatively or 'auto_0.1' to give init value
    'target_update_interval': 5,  # Default: 1
    'target_entropy': 0.1,  # Default: 'auto', use float alternatively

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'large',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
        'n_critics': 3,  # Default: 2
        'share_features_extractor': True,  # Default: False
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
        }, f)


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='sac',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=None,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    create_stats_file(SAVE_PATH, EXP_PARAMS)