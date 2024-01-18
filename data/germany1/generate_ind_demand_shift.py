import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(22)

data = pd.read_csv('54.2000_8.9000_all_data.csv', index_col=0)
data['Date_local'] = pd.to_datetime(data['Date_local'])

# print(data['Date'][5].week)

workday = [0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.21, 0.36, 0.68, 0.74, 0.81, 0.89, 0.74, 0.85, 0.93, 0.88, 0.8, 0.61, 0.31,
           0.2, 0.14, 0.1, 0.1, 0.1]
weekend = [0.1 for i in range(24)]
holidays = ['2021-12-31', '2022-01-01', '2022-04-15', '2022-04-18', '2022-05-01', '2022-05-26',
            '2022-06-06', '2022-10-03', '2022-10-31', '2022-12-25', '2022-12-26', '2022-12-31']
factor = 4_500

week = 0
weekly_perturb = 1  # overwritten in loop below
demands = []
for index, row in data.iterrows():
    dt = row['Date_local']
    # print(dt)
    # print(f'Weekday: {dt.weekday()} | Hour: {dt.hour}')
    # Allocate proper profile
    if dt.weekday() == 5 or dt.weekday() == 6 or str(dt.date()) in holidays:  # 5 = Saturday, 6 = Sunday
        demand = weekend[dt.hour]
    else:
        demand = workday[dt.hour]

    # Adjustment (demand increase) due to temperature
    temp = row['temperature']
    # HEAT
    if temp > 25:
        demand += 0.01
        if temp > 30:
            demand += 0.025
    # COLD
    if temp < 10:
        demand += 0.01
        if temp < 0:
            demand += 0.025
            if temp < -10:
                demand += 0.05
                if temp < -20:
                    demand += 0.05

    # Random perturbation
    # Daily
    demand *= np.random.normal(loc=1, scale=0.05)
    # Weekly
    if week != dt.isocalendar().week:
        week = dt.isocalendar().week
        weekly_pertub = np.random.normal(loc=1, scale=0.1)
    demand *= weekly_perturb

    demand *= factor
    demands.append(demand)

    # if index == 744:
    #     break

# plt.plot(demands, label='created demand')
# plt.ylim(-1000, 45000)
# plt.legend()
# plt.xlabel('Hours')
# plt.ylabel('Demand (kW)')
# plt.title('Shift demand')
# plt.show()

df = pd.Series(demands)
df.name = 'demand'
print(df.mean())

df.to_csv('ind_demand_shift.csv')
