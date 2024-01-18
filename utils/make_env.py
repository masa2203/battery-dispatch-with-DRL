from typing import Optional, Dict, Any

import gym.wrappers
import stable_baselines3.common.vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from utils.wrappers import *


def make_env(env,
             env_kwargs: dict,
             tracking: bool = False,
             allow_early_resets: bool = True,
             path: Optional[str] = None,
             seed: int = 22,
             inactive_penalty: Optional[Dict[str, Any]] = None,
             action_corr_penalty: Optional[int] = None,
             soc_penalty: Optional[int] = None,
             flatten_obs: bool = True,
             discrete_actions: Optional[list] = None,
             frame_stack: Optional[int] = None,
             norm_obs: bool = True,
             norm_reward: bool = True,
             gamma: float = 0.99,
             ):
    e = Monitor(env=env(**env_kwargs, tracking=tracking),
                allow_early_resets=allow_early_resets,  # allow finish rollout for PPO -> throws error otherwise
                filename=path)
    e.seed(seed)

    if inactive_penalty is not None:  # Must be called before flattening
        e = InactivityPenalty(e, **inactive_penalty)

    if action_corr_penalty is not None:
        e = ActionCorrectionPenalty(e, penalty=action_corr_penalty)

    if soc_penalty is not None:
        e = SOCPenalty(e, penalty=soc_penalty, max_soc=env_kwargs['storage']['max_soc'])

    if flatten_obs:
        e = gym.wrappers.FlattenObservation(e)

    # Add discrete action wrapper
    if discrete_actions is not None:
        e = DiscreteActions(e, discrete_actions)

    e = DummyVecEnv([lambda: e])

    # Stack observation
    if frame_stack is not None:
        e = stable_baselines3.common.vec_env.VecFrameStack(e, n_stack=frame_stack)

    e = VecNormalize(e, norm_obs=norm_obs, norm_reward=norm_reward, gamma=gamma)

    return e
