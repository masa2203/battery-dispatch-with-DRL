import math
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)

df = pd.read_csv('2022_electricity_final.csv',
                 index_col=0,
                 parse_dates=['Date'],
                 # cache_dates=True,
                 )

# ADD HOURLY FEATURES USING SIN AND COS
df['Date_local'] = pd.to_datetime(df['Date']).dt.tz_convert('America/Edmonton')
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

df.drop(['h_norm'], axis=1, inplace=True)
df.drop(['w_norm'], axis=1, inplace=True)
df.drop(['m_norm'], axis=1, inplace=True)
df.to_csv('alberta_2022_electricity_final.csv')
