import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz
import math

# pd.set_option('display.max_columns', None)


df = pd.read_csv('2022raw.csv', parse_dates=[['date', 'hour']], dayfirst=True)
df = df.rename(columns={'date_hour': 'Date'})
print(df)
# print(df.dtypes)
# print(df[280:300]['Date'])

# Daylight saving time
# -----------------------------------------------------------------

# print(df[2038:2049]['Date'])
# print(df[7244:7255]['Date'])

my_timezone = pytz.timezone('Europe/Berlin')

dst_dict = {
    2022: ('2022-03-27 03:00:00', '2022-10-30 02:00:00'),  # DST dates for 2022: March 13 & Nov 06
}

to_dst_index = df.index[df['Date'] == dst_dict[2022][0]].tolist()  # get indices of DST
to_dt_index = df.index[df['Date'] == dst_dict[2022][1]].tolist()  # and DT

# df.at[to_dst_index[0], 'Date'] = df['Date'][to_dst_index[0]] - timedelta(hours=1)  # To DST in March
# df.at[to_dt_index[0], 'Date'] = df['Date'][to_dt_index[0]] - timedelta(hours=1)  # To DT in November

# Create array indicating DT or DST
infer_dst = np.array([True] * df.shape[0])  # False is considered DT, True is considered DST
infer_dst[:to_dst_index[0]] = False
infer_dst[to_dt_index[1]:] = False

# Convert to local timezone with correct DLS and then to UTC
dl = df['Date'].dt.tz_localize(my_timezone, ambiguous=infer_dst)
df.insert(loc=1, column='Date_local', value=dl)
df['Date'] = df['Date_local'].dt.tz_convert(pytz.utc)

# print(df[2038:2049]['Date'])
# print(df[7244:7255]['Date'])
# print(df[7244:7255]['Date_local'])

df = df[df['Date'].dt.year == 2022]

# CONVERT PRICES TO CANADIAN DOLLARS
euro_to_cad = 1.45  # corresponds roughly to 5-year average
df['pool_price'] *= euro_to_cad

# ADD HOURLY FEATURES USING SIN AND COS
df['h'] = df['Date_local'].dt.hour

df['h_norm'] = 2 * math.pi * df['Date_local'].dt.hour / 23  # 23 as max of column
df["cos_h"] = np.cos(df["h_norm"])
df["sin_h"] = np.sin(df["h_norm"])

# ADD WEEKLY FEATURES USING SIN AND COS
df['week'] = df['Date_local'].dt.week
df['w_norm'] = 2 * math.pi * df['Date_local'].dt.week / 52  # 52 as max of column
df["cos_w"] = np.cos(df["w_norm"])
df["sin_w"] = np.sin(df["w_norm"])

# ADD MONTHLY FEATURES USING SIN AND COS
df['month'] = df['Date_local'].dt.month
df['m_norm'] = 2 * math.pi * df['Date_local'].dt.month / 12  # 12 as max of column
df["cos_m"] = np.cos(df["m_norm"])
df["sin_m"] = np.sin(df["m_norm"])

# ADD BINARY WORKDAY FEATURE
# list stems from demand generation file
holidays = ['2021-12-31', '2022-01-01', '2022-04-15', '2022-04-18', '2022-05-01', '2022-05-26',
            '2022-06-06', '2022-10-03', '2022-10-31', '2022-12-25', '2022-12-26', '2022-12-31']

df['workday'] = [0 if d.dayofweek == 5 or d.dayofweek == 6 or str(d.date()) in holidays else 1 for d in df['Date_local']]

df.drop(['h_norm'], axis=1, inplace=True)
df.drop(['w_norm'], axis=1, inplace=True)
df.drop(['m_norm'], axis=1, inplace=True)
print(df)

df.to_csv('de_2022_electricity_final.csv')
