import os
import json
from typing import Any

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

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
SAVE_PATH = os.path.join('../log/', ENV_KWARGS['env_name'], 'ddpg/run', input('Save in folder: ')) \
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

# DDPG PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'learning_rate': 1e-3,  # Default: 1e-3
    # 'learning_rate': linear_schedule(1e-3),
    'buffer_size': 1_000_000,  # Default: 1e6
    'learning_starts': 100,  # Default: 100
    'batch_size': 100,  # Default: 100
    'tau': 0.005,  # Default: 0.005
    'gamma': 0.99,  # Default: 0.99
    # 'train_freq': 100,  # Default: (1, 'episode')
    'train_freq': (1, 'episode'),  # Default: (1, 'episode'), integer treated as step count
    'gradient_steps': 5,  # Default: -1 (-1 = as many steps as rollout)
    'action_noise': None,  # Default: None
    # 'action_noise': OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.01*np.ones(1)),  # Default: None

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'tiny',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
        'n_critics': 2,  # Default: 2
        'share_features_extractor': True,  # Default: True
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
        agent='ddpg',
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
