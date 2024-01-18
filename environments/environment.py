from typing import Optional

import numpy as np
import pandas as pd
import gym
from gym import spaces

from environments.grid_model import GridModel
from environments.storage_model import BESS, DODDegradingBESS


class BESS_Base_Env(gym.Env):
    """
    Environment class based on gym, with battery as dispatched component.

    Base class with core functionality.
    """

    def __init__(self,
                 env_name: str,
                 storage: dict,
                 resolution_h: Optional[float] = 1.0,
                 modeling_period_h: Optional[int] = 8760,
                 tracking: Optional[bool] = True):
        super(BESS_Base_Env, self).__init__()

        self.env_name = env_name
        self.storage_dict = storage
        self.resolution_h = resolution_h
        self.modeling_period_h = modeling_period_h
        self.tracking = tracking

        # Storage
        self.storage = None
        self.storage_flow = None  # required for ActionCorrectionPenalty wrapper

        self.count = 0
        self.obs = None

        # Tracking
        self.actions = []
        self.rewards = []
        self.e_balances = []  # deficit or surplus power
        self.env_log = None

    def reset(self):
        self.count = 0
        self.actions = []
        self.rewards = []
        self.e_balances = []  # deficit or surplus power
        self.storage = self._init_storage()
        obs, done = self._get_obs()
        return obs

    def step(self, action):
        print('Base class method called')

    def render(self, **kwargs):
        pass

    def _get_obs(self):
        print('Base class method called')
        return None, False

    def _init_storage(self):
        """Initialize storage."""
        if self.storage_dict['degradation'] is None:
            storage = BESS(**self.storage_dict, resolution_h=self.resolution_h, tracking=self.tracking)
        elif self.storage_dict['degradation']['type'] == 'DOD':
            storage = DODDegradingBESS(**self.storage_dict, resolution_h=self.resolution_h, tracking=self.tracking)
        else:
            raise ValueError('Unknown storage type called!')
        return storage

    def _tracking(self, action, reward, e_balance=None):
        """Keep track of storage behavior over time."""
        self.actions.append(action)
        self.rewards.append(reward)
        if e_balance is not None:
            self.e_balances.append(e_balance)  # e-delivered - demand

    def _get_info(self):
        # Add some env info and return as dict
        return {'soc': self.storage.soc}

    def _get_episode_info(self):
        return {
            'actions': self.actions,
            'rewards': self.rewards,
            'e_balances': self.e_balances,
            'socs': self.storage.socs,
            'bes_energy_flows': self.storage.energy_flows,
            'degr_costs': self.storage.degr_costs,
        }

    def return_episode_info(self):
        return self.env_log

    def start_tracking(self):
        self.tracking = True

    @staticmethod
    def _clip_action(a, up, low):
        """Clips action to range [up,low]"""
        if a > up:
            a = up
        if a < low:
            a = low
        return float(a)


class HES_BESS_Env1(BESS_Base_Env):
    """
    Environment class based on gym, with battery as dispatched component.

    Battery is part of a hybrid energy system with renewables.
    """

    def __init__(self, env_name: str,
                 data_file: str, demand_file: str, state_vars: list,
                 sell_surplus: bool, e_price_fix_fee: float,
                 storage: dict, num_wt: int, pv_cap_mw: float,
                 resolution_h: float = 1.0, modeling_period_h: int = 8760,
                 tracking: bool = True):
        super().__init__(env_name, storage, resolution_h, modeling_period_h, tracking)

        self.state_vars = state_vars

        self.data = pd.read_csv(data_file, index_col=0)
        self.data['re_power'] = np.nan
        # Adjust columns for PV and wind to component sizes
        self.data['wind_power'] *= num_wt
        self.data['pv_power'] *= pv_cap_mw
        self.data['re_power'] = self.data['pv_power'] + self.data['wind_power'] / 1000

        self.data.drop([i for i in self.data.columns if i != 'Date' and i not in self.state_vars],
                       axis=1, inplace=True)

        demand = pd.read_csv(demand_file, index_col=0)
        self.data['demand'] = demand['demand'] / 1000  # add demand column to data-file, convert to MW
        demand_min, demand_max = self.data['demand'].min(), self.data['demand'].max()  # get min/max of demand

        # DEFINE OBS SPACE
        self.observation_space = spaces.Dict(
            {
                'demand': spaces.Box(low=demand_min, high=demand_max, shape=(1,)),
                'soc': spaces.Box(low=0, high=1, shape=(1,))
            }
        )
        for clm in self.state_vars:  # add variables from data file to observation space
            low, high = self.data[clm].min(), self.data[clm].max()
            self.observation_space[clm] = spaces.Box(low=low, high=high, shape=(1,))

        # Initialize components
        # Grid
        self.grid = GridModel(demand_profile=True,
                              ind_demand=True,
                              sell_surplus=sell_surplus,
                              e_price_fix_fee=e_price_fix_fee)

        # DEFINE ACTION SPACE
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

    def step(self, action):
        """Execute one step in the environment."""
        if np.isnan(action).any():
            raise ValueError('Action is NAN!')
        action = self._clip_action(action[0], self.action_space.high, self.action_space.low)

        self.storage_flow, degr_cost = self.storage.step(action=action,
                                                         avail_power=self.obs['re_power'].item())

        reward = self._compute_reward(action=action,
                                      power=self.obs['re_power'],
                                      degr_cost=degr_cost,
                                      storage_flow=self.storage_flow)

        if np.isnan(reward):
            raise ValueError('Reward is NAN!')

        self.count += 1
        next_obs, done = self._get_obs()

        return next_obs, reward, done, self._get_info()

    def _compute_reward(self, action, power, degr_cost, storage_flow):
        """Compute the reward."""
        # Add/subtract power going to or coming from storage
        e_delivered = max(power + storage_flow, 0)

        # Compute sales of electricity
        cash_flow = self.grid.get_grid_interaction(e_delivered, self.obs)

        reward = (cash_flow - degr_cost).item()

        if self.tracking:
            self._tracking(round(action, 3),
                           round(reward, 3),
                           round((e_delivered - self.obs['demand']).item(), 3))

        return reward

    def _get_obs(self):
        if self.count == self.data.shape[0]:  # Check termination
            self.env_log = self._get_episode_info()
            return self.obs, True

        row = self.data.iloc[self.count]
        obs = {'demand': np.array([row['demand']], dtype=np.float32),
               'soc': np.array([self.storage.soc], dtype=np.float32)}
        for i in self.state_vars:
            obs[i] = np.array([row[i]], dtype=np.float32)

        self.obs = obs
        return obs, False


class EA_BESS(BESS_Base_Env):
    """
    Environment class based on gym, with battery as dispatched component.

    Single battery working on energy arbitrage.
    """

    def __init__(self, env_name: str,
                 data_file: str, state_vars: list,
                 storage: dict, e_price_fix_fee: float,
                 resolution_h: float = 1.0, modeling_period_h: int = 8760,
                 tracking: bool = True):
        super().__init__(env_name, storage, resolution_h, modeling_period_h, tracking)

        self.state_vars = state_vars

        self.data = pd.read_csv(data_file, index_col=0)

        self.data.drop([i for i in self.data.columns if i != 'Date' and i not in self.state_vars],
                       axis=1, inplace=True)

        # DEFINE OBS SPACE
        self.observation_space = spaces.Dict(
            {
                'soc': spaces.Box(low=0, high=1, shape=(1,))
            }
        )
        for clm in self.state_vars:  # add variables from data file to observation space
            low, high = self.data[clm].min(), self.data[clm].max()
            self.observation_space[clm] = spaces.Box(low=low, high=high, shape=(1,))

        # Initialize components
        # Grid
        self.grid = GridModel(demand_profile=False,
                              ind_demand=False,
                              e_price_fix_fee=e_price_fix_fee)

        # DEFINE ACTION SPACE
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

    def step(self, action):
        """Execute one step in the environment."""
        if np.isnan(action).any():
            raise ValueError('Action is NAN!')
        action = self._clip_action(action[0], self.action_space.high, self.action_space.low)

        self.storage_flow, degr_cost = self.storage.step(action=action,
                                                         avail_power=100)

        reward = self._compute_reward(action=action,
                                      degr_cost=degr_cost,
                                      storage_flow=self.storage_flow)

        if np.isnan(reward):
            raise ValueError('Reward is NAN!')

        self.count += 1
        next_obs, done = self._get_obs()

        return next_obs, reward, done, self._get_info()

    def _compute_reward(self, action, storage_flow, degr_cost):
        """Compute the reward."""
        # Compute sales of electricity
        cash_flow = self.grid.get_grid_interaction(storage_flow, self.obs)

        reward = (cash_flow - degr_cost).item()

        if self.tracking:
            self._tracking(round(action, 3),
                           round(reward, 3),
                           None)  # None as no demand

        return reward

    def _get_obs(self):
        if self.count == self.data.shape[0]:  # Check termination
            self.env_log = self._get_episode_info()
            return self.obs, True

        row = self.data.iloc[self.count]
        obs = {'soc': np.array([self.storage.soc], dtype=np.float32)}
        for i in self.state_vars:
            obs[i] = np.array([row[i]], dtype=np.float32)

        self.obs = obs
        return obs, False
