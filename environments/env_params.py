# PARAMETER FILE FOR ENV

# BATTERY DEGRADATION
dod_degr = {
    'type': 'DOD',
    'battery_capex': 300_000,  # CAD/MWh
    'k_p': 1.14,  # Peukert lifetime constant, degradation parameter
    'N_fail_100': 6_000,  # number of cycles at DOD=1 until battery is useless
    'add_cal_age': False,  # adds fixed cost for calendar ageing if True via MAX-operator
    'battery_life': 20,  # expected battery life in years
}

# PLANT - ALBERTA - ENERGY ARBITRAGE - 2022
al4_bat_ea = {
    'env_name': 'al4_bat_ea',  # used for saving path
    'data_file': '../data/alberta3/alberta_2022_electricity_final.csv',
    'state_vars': ['pool_price'],  # list of data columns to serve as state var
    'e_price_fix_fee': 10.0,  # CAD per MWh, added to electricity price
    'storage': dict(total_cap=10,  # MWh
                    max_soc=0.8,  # fraction of total capacity
                    min_soc=0.2,  # fraction of total capacity
                    max_charge_rate=2.5,  # MW
                    max_discharge_rate=2.5,  # MW
                    charge_eff=0.92,  # fraction
                    discharge_eff=0.92,  # fraction
                    aux_equip_eff=1.0,  # fraction, applied to charge & discharge
                    self_discharge=0.0,  # fraction, applied to every step (0 = no self-discharge)
                    init_soc=0.5,  # in MWh or None for randomly selected
                    degradation=dod_degr,
                    ),
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}

# PLANT - GERMANY - HES - 2022
de1_bat_hes = {
    'env_name': 'de1_bat_hes',  # used for saving path
    'data_file': '../data/germany1/54.2000_8.9000_all_data.csv',
    'demand_file': '../data/germany1/ind_demand_shift.csv',
    'state_vars': ['pool_price', 're_power'],  # list of data columns to serve as state var
    'sell_surplus': False,
    'e_price_fix_fee': 10.0,  # CAD per MWh, added to electricity price
    'storage': dict(total_cap=20,  # MWh
                    max_soc=0.8,  # fraction of total capacity
                    min_soc=0.2,  # fraction of total capacity
                    max_charge_rate=2.5,  # MW
                    max_discharge_rate=2.5,  # MW
                    charge_eff=0.92,  # fraction
                    discharge_eff=0.92,  # fraction
                    aux_equip_eff=1.0,  # fraction, applied to charge & discharge
                    self_discharge=0.0,  # fraction, applied to every step (0 = no self-discharge)
                    init_soc=0.5,  # in MWh or None for randomly selected
                    degradation=dod_degr,
                    ),
    'num_wt': 1,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 1,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}
