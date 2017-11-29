import pandas as pd

user_log_v1 = pd.read_csv('../input/processed_user_log.csv', nrows=1000)
user_log_v2 = pd.read_csv('../input/processed_user_log_v2.csv', nrows=1000)

user_log = user_log_v1.append(user_log_v2, ignore_index=True)

print(user_log.head())
