import gym
import numpy as np


class DiscreteActions(gym.ActionWrapper):
    """Discretize action space."""
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]


class InactivityPenalty(gym.RewardWrapper):
    """Penalize agent for not using battery"""
    def __init__(self, env, patience, penalty):
        super().__init__(env)
        self.patience = patience
        self.penalty = penalty
        self.old_soc = 0.5
        self.count = 0

    def reward(self, reward):
        if self.old_soc == self.env.obs['soc']:
            self.count += 1
            if self.count >= self.patience:
                reward -= self.penalty
        else:
            self.count = 0
        self.old_soc = self.env.obs['soc']
        return reward


class SOCPenalty(gym.RewardWrapper):
    """Penalize agent for low SOCs."""
    def __init__(self, env, penalty, max_soc=0.8):
        super().__init__(env)
        self.penalty = penalty
        self.max_soc = max_soc

    def reward(self, reward):
        p = (self.max_soc - self.env.obs['soc']).item() * self.penalty
        reward -= p
        return reward


class ActionCorrectionPenalty(gym.RewardWrapper):
    """Penalize agent if action correction through safety layer is required."""
    def __init__(self, env, penalty):
        super().__init__(env)
        self.penalty = penalty
        self.sd = self.env.storage_dict
        self.last_action = None
        self.actual_action = None

    def step(self, action):
        self.last_action = action.item()
        return super().step(action)

    def reward(self, reward):
        storage_flow = self.env.storage_flow
        if storage_flow > 0:  # discharge
            efficiencies = self.sd['discharge_eff'] * self.sd['aux_equip_eff'] * (1 - self.sd['self_discharge'])
            self.actual_action = storage_flow / efficiencies / self.sd['max_discharge_rate']
        elif storage_flow < 0:  # charge
            # Note: efficiencies are reflected through lower SOC after charging -> not required to consider here
            self.actual_action = storage_flow / self.sd['max_charge_rate']
        else:
            self.actual_action = 0

        diff = round(np.abs(self.last_action - self.actual_action), 4)
        reward -= np.square(diff) * self.penalty
        return reward
