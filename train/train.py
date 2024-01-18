import os
import time
import json
from typing import Optional

from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from utils.callbacks import ProgressBarManager
from utils.make_env import make_env
from utils.utils import get_env_log_data


AGENTS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'dqn': DQN,
    'ddpg': DDPG,
}


def train_rl_agent(
        agent: str,
        run: int,
        path: Optional[str],
        exp_params: dict,
        env_id,
        env_kwargs: dict,
        rl_params: dict,
        verbose: int = 0,
        discrete_actions: Optional[list] = None,
        logger_type: Optional[list] = None
):

    agent = AGENTS[agent]

    if logger_type is None:
        logger_type = ['csv']  # Default: save progress.csv file if path is not None

    tt = int(exp_params['n_episodes'] * 8760)  # total timesteps

    run_path = os.path.join(path, f'run_{run}') if path is not None else None

    if path is not None:
        os.makedirs(run_path, exist_ok=True)

    start = time.time()
    seed = exp_params['seed'] + run
    print('|| Run #{} | Seed #{} ||'.format(run, seed))

    # CREATE ENVIRONMENT
    env = make_env(env=env_id,
                   env_kwargs=env_kwargs,
                   path=os.path.join(run_path, "train_monitor.csv") if path is not None else None,
                   seed=seed,
                   inactive_penalty=exp_params['inactive_penalty'],
                   action_corr_penalty=exp_params['action_corr_penalty'],
                   flatten_obs=exp_params['flatten_obs'],
                   discrete_actions=discrete_actions,
                   frame_stack=exp_params['frame_stack'],
                   norm_obs=exp_params['norm_obs'],
                   norm_reward=exp_params['norm_reward'],
                   gamma=rl_params['gamma'],
                   )

    # DEFINE MODEL
    model = agent(env=env, verbose=verbose, seed=seed, **rl_params)
    if path is not None:
        logger = configure(run_path, logger_type)
        model.set_logger(logger)
    with ProgressBarManager(total_timesteps=tt) as callback:
        # Evaluation during training
        if exp_params['eval_while_training']:
            eval_env = make_env(env=env_id,
                                env_kwargs=env_kwargs,
                                path=os.path.join(run_path, 'eval_monitor.csv') if path is not None else None,
                                seed=seed,
                                inactive_penalty=None,
                                action_corr_penalty=None,
                                flatten_obs=exp_params['flatten_obs'],
                                discrete_actions=discrete_actions,
                                frame_stack=exp_params['frame_stack'],
                                norm_obs=exp_params['norm_obs'],
                                norm_reward=exp_params['norm_reward'],
                                gamma=rl_params['gamma'],
                                )

            eval_callback = EvalCallback(eval_env=eval_env,
                                         n_eval_episodes=1,
                                         eval_freq=exp_params['eval_freq'],
                                         deterministic=True,
                                         # best_model_save_path=run_path,
                                         best_model_save_path=None,
                                         verbose=verbose)

            model.learn(total_timesteps=tt, callback=[callback, eval_callback])
        # Evaluation only after training
        else:
            model.learn(total_timesteps=tt, callback=callback)

    env.training = False
    env.norm_reward = False
    env.env_method('start_tracking')  # Start tracking env variables for evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    if run_path is not None:
        # model.save(os.path.join(run_path, 'model'))  # Save best model instead through callback
        # env.save(os.path.join(run_path, 'env.pkl'))
        log_data = get_env_log_data(env=env, mean_reward=mean_reward, start_time=start)
        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)

    env.close()

    print()
    print(f'Execution time = {time.time() - start}s')
