"""
Manuel Sage, August 2022

Class to model electricity grid and its interaction with the plant
"""
import warnings


class GridModel:
    """Class to model the grid and its interaction with the plant"""
    def __init__(self,
                 demand_profile: bool = False,  # use demand or not
                 ind_demand: bool = False,  # treat demand as industrial load (only active with demand_profile = True)
                 sell_surplus: bool = False,  # if set to True, surplus power is sold (industrial case only)
                 e_price_fix_fee: float = 0.0,
                 ):
        """
        :param demand_profile: If False, all power is sold at current pool price
        :type demand_profile: bool
        :param ind_demand: If True, demand profile is treated as industrial demand, else as grid demand.
        Only works with demand_profile = True
        :type ind_demand: bool
        :param sell_surplus: Only relevant if ind_demand = True, allows to sell excess power
        :type sell_surplus: bool
        :param e_price_fix_fee: CAD per MWh, added to pp AFTER multiplying with var fee
        :type e_price_fix_fee: float
        """
        self.demand_profile = demand_profile
        self.ind_demand = ind_demand
        self.sell_surplus = sell_surplus
        self.e_price_fix_fee = e_price_fix_fee

    def get_grid_interaction(self, e_flow, obs):
        """Models grid response to produced electricity.

        Handles two cases:
        I: All power is sold at given pool price.
        II: Demand by industry, optional to sell surplus, producing less causes electricity import at pool price.

        :param e_flow: all energy generated by the plant in current time-step, in MWh
        :type e_flow: float
        :param obs: row in data file corresponding to current hour
        :type obs: pandas series
        :return: cashflow (negative if power was imported) in CAD, penalty for missing demand
        :rtype: tuple (float, float)
        """
        pp = obs['pool_price']
        # print('Pool price: ', pp)

        # CASE I: No demand, sell all electricity to grid
        if not self.demand_profile:
            modified_pp = pp + self.e_price_fix_fee
            cash_flow = e_flow * modified_pp
            return cash_flow

        assert e_flow >= 0, 'Produced power cannot be negative!'

        # CASE II: Demand by industrial facility (deficit is bought from grid, surplus is sold optionally)
        if self.demand_profile and self.ind_demand:
            demand = obs['demand']
            assert demand >= 0, 'Demand cannot be negative!'

            if self.sell_surplus and e_flow > demand:  # sell excess power
                cash_flow = (e_flow - demand) * pp
            elif e_flow < demand:  # purchase deficit power
                modified_pp = pp + self.e_price_fix_fee
                cash_flow = (e_flow - demand) * modified_pp
            else:
                cash_flow = 0
            return cash_flow
