import random
import warnings
import numpy as np


class BESS:
    def __init__(self, total_cap: int,
                 max_soc: float, min_soc: float,
                 max_charge_rate, max_discharge_rate,
                 charge_eff: float, discharge_eff: float,
                 aux_equip_eff: float = 1.0, self_discharge: float = 0.0,
                 init_soc: float = 0.5,
                 resolution_h: float = 1.0,
                 degradation=None,
                 tracking: bool = True,
                 ):
        """
        Construct a new BESS object.

        :total_cap: Total capacity of the storage (MWh)
        :max_soc: Maximum allowed state of charge of total capacity (fraction)
        :min_soc: Minimum allowed state of charge of total capacity (fraction)
        :max_charge_rate: Maximum possible rate of charge (MW)
        :max_discharge_rate: Maximum possible rate of discharge (MW) NEGATIVE!
        :charge_eff: Charging efficiency (fraction)
        :discharge_eff: Discharging efficiency (fraction)
        :aux_equip_eff: Efficiency of auxiliary equipment applied to charge and discharge cycles (fraction)
        :self_discharge: discharge of storage, applied at every time step (fraction)
        :init_soc: initial state of charge, selected randomly if not specified (fraction)
        :resolution_h: resolution in hours
        :degradation: if ageing is considered
        :tracking: track variables in step function
        """
        # ARGUMENTS
        self.total_cap = total_cap  # MWh
        self.max_soc = max_soc  # fraction of total capacity
        self.min_soc = min_soc  # fraction of total capacity
        self.max_charge_rate = max_charge_rate  # MW
        self.max_discharge_rate = max_discharge_rate  # MW
        self.charge_eff = charge_eff  # fraction
        self.discharge_eff = discharge_eff  # fraction
        self.aux_equip_eff = aux_equip_eff  # fraction, applied to charge & discharge
        self.self_discharge = self_discharge  # fraction, applied to every step
        self.resolution_h = resolution_h  # resolution in hours
        self.degradation = degradation
        self.tracking = tracking

        if init_soc is not None:
            self.soc = init_soc
        else:
            self.soc = random.randrange(0, self.total_cap) / self.total_cap

        # TRACKERS
        self.socs = []  # tracks SOCs
        self.energy_flows = []  # tracks energy flows from plant view
        self.degr_costs = []  # tracks degradation cost

    def step(self, action, avail_power):
        """
        Conduct one step with the storage.

        :action: range(-1,1) used to charge or discharge the storage, negative sign for charge
        :avail_power: max. power available for charging (MW)
        """

        energy_flow = self._soc_change(action, avail_power)
        degr_cost = 0

        if self.tracking:
            self._tracking(energy_flow, degr_cost)

        return energy_flow, degr_cost

    def _soc_change(self, action, avail_power):
        """
        Change SOC according to action, i.e., either charging, discharging, or keeping unchanged.

        :action: range(-1,1) used to charge or discharge the storage, negative sign for charge
        :avail_power: max. power available for charging (MW)
        """
        if action > 1 or action < -1:
            warnings.warn('WARNING: Storage charge rates outside interval (-1,1) !')

        # Self discharge of storage
        self.soc = self.soc * (1 - self.self_discharge)

        # Convert action into charge/discharge rate and get energy flow (variable for actual energy exchange)
        if action > 0:  # discharge, battery gives energy
            rate = self.max_discharge_rate * action
            energy_flow = self._discharge(rate)
        elif action < 0:  # charge, battery takes energy
            rate = self.max_charge_rate * action
            energy_flow = self._charge(rate, avail_power)
        else:
            energy_flow = 0

        return energy_flow  # negative value for charging / positive value for discharging

    def _charge(self, rate, avail_power):
        """
        Charges the storage.

        Notes:
        - Charges with desired rate unless less power is available
        - Charges with desired rate unless less max SOC is reached
        - Efficiencies are reflected through lower SOC values after charging

        :param rate: charge rate in MW
        :param avail_power: available power for charging in MW
        :return: Energy used to charge the storage
        """
        rate = -rate  # Charge rate is negative -> make positive
        # Pick min of desired charge rate and available power, multiply by time to obtain energy
        available_energy = min(rate, avail_power) * self.resolution_h

        # Update SOC
        old_soc = self.soc
        new_soc = self.soc + (available_energy * self.charge_eff * self.aux_equip_eff) / self.total_cap
        self.soc = min(new_soc, self.max_soc)  # avoids overcharging

        # Compute effective energy flow
        energy_flow = (old_soc - self.soc) * self.total_cap / (self.charge_eff * self.aux_equip_eff)

        return energy_flow

    def _discharge(self, rate):
        """
        Discharges the storage.

        Notes:
        - Charges with desired rate unless min SOC is reached
        - Efficiencies are reflected through less energy flow compared to SOC reduction

        :param rate: discharge rate in MW (negative)
        :return: Energy drawn from storage (as available for use)
        """

        # Update SOC, no efficiencies here
        old_soc = self.soc
        new_soc = self.soc - (rate * self.resolution_h) / self.total_cap
        self.soc = max(new_soc, self.min_soc)

        # Compute effective energy flow, factor in losses during discharge
        energy_flow = (old_soc - self.soc) * self.discharge_eff * self.aux_equip_eff * self.total_cap

        return energy_flow

    def _tracking(self, energy_flow, degr_cost):
        """Keep track of storage behavior over time."""
        self.socs.append(round(self.soc, 2))
        self.energy_flows.append(round(energy_flow, 2))
        self.degr_costs.append(round(degr_cost, 2))


class DODDegradingBESS(BESS):
    """
    Add battery ageing cost based on DOD changes.

    Based on the work of Yi Dong et al. (2021) and others cited in the paper:
    https://www.sciencedirect.com/science/article/pii/S0378779621002108?via%3Dihub
    """

    def __init__(self, total_cap: int,
                 max_soc: float, min_soc: float,
                 max_charge_rate, max_discharge_rate,
                 charge_eff: float, discharge_eff: float,
                 aux_equip_eff: float, self_discharge: float,
                 init_soc: float,
                 resolution_h: float,
                 degradation: dict,
                 tracking: bool):
        super().__init__(total_cap, max_soc, min_soc, max_charge_rate, max_discharge_rate,
                         charge_eff, discharge_eff, aux_equip_eff, self_discharge, init_soc,
                         resolution_h, tracking=tracking)

        self.investment_cost = self.total_cap * degradation['battery_capex']
        self.k_p = degradation['k_p']
        self.N_fail_100 = degradation['N_fail_100']
        self.add_cal_age = degradation['add_cal_age']
        self.battery_life = degradation['battery_life']

    def step(self, action, avail_power):
        """
        Conduct one step with the storage.

        :action: range(-1,1) used to charge or discharge the storage, negative sign for charge
        :avail_power: max. power available for charging (MW)
        """
        soc_old = self.soc
        energy_flow = self._soc_change(action, avail_power)
        degr_cost = self._get_degr_cost(self.soc, soc_old)

        if self.tracking:
            self._tracking(energy_flow, degr_cost)

        return energy_flow, degr_cost

    def _get_degr_cost(self, soc, soc_old):
        """
        Compute degradation cost based on depth of discharge.

        :param soc: current SOC [0,1]
        :param soc_old: SOC of last time step [0,1]
        :return: degradation cost
        """
        denominator = 2 * self.N_fail_100
        numerator = np.abs((1 - soc) ** self.k_p - (1 - soc_old) ** self.k_p)
        degr_cost = self.investment_cost * numerator / denominator

        # Add calendar age if desired
        if self.add_cal_age:
            degr_cost = max(degr_cost, self.investment_cost / self.battery_life / 8760)
        return degr_cost
