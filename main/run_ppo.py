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
DISCRETE_ACTIONS = None
ENV_KWARGS['state_vars'] = ['pool_price', 'cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m']

# DE-LF/REU
# ENV = HES_BESS_Env1
# ENV_KWARGS = de1_bat_hes
# DISCRETE_ACTIONS = None
# ENV_KWARGS['state_vars'] = ['pool_price', 're_power', 'cos_h', 'sin_h', 'cos_w', 'sin_w', 'cos_m', 'sin_m']
# -----------------------------------------------------------------

# LOG
CREATE_LOG = False
VERBOSE = 0
LOGGER_TYPE = ["csv", "tensorboard"]
SAVE_PATH = os.path.join('../log/', ENV_KWARGS['env_name'], 'ppo/run', input('Save in folder: ')) \
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
    'frame_stack': None,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': 8760 * 1,
}

# PPO PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'device': 'cpu',
    # 'learning_rate': 3e-4,  # Default: 3e-4
    'learning_rate': linear_schedule(8.4e-4),  # Default: 3e-4
    'n_steps': 512,  # Default: 2048
    'batch_size': 512,  # Default: 64
    'n_epochs': 10,  # Default: 10
    'gamma': 0.993859,  # Default: 0.99
    'gae_lambda': 0.93767,  # Default: 0.95
    'clip_range': linear_schedule(0.2),  # Default: 0.2
    'clip_range_vf': linear_schedule(0.3),  # Default: None
    'normalize_advantage': True,  # Default: True
    'ent_coef': 0.249023,  # Default: 0.0
    'vf_coef': 0.575101,  # Default: 0.5
    'max_grad_norm': 0.5775,  # Default: 0.5

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'medium',  # Default: None
        'activation_fn': 'relu',  # Default: tanh
        'ortho_init': False,  # Default: True
        'squash_output': False,  # Default: False
        'share_features_extractor': False,  # Default: True
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': str(DISCRETE_ACTIONS),
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
        }, f)


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='ppo',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=DISCRETE_ACTIONS,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    create_stats_file(SAVE_PATH, EXP_PARAMS)
