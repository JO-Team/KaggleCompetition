import pandas as pd

user_log_v1 = pd.read_csv('../input/processed_user_log.csv')
print(len(user_log_v1))
user_log_v2 = pd.read_csv('../input/processed_user_log_v2.csv')
print(len(user_log_v2))

user_log = user_log_v1.append(user_log_v2, ignore_index=True)

func = {'date_min': ['min'], 'date_max': ['max'], 'date_count': ['count'],
        'num_25_sum': ['sum'], 'num_50_sum': ['sum'],
        'num_75_sum': ['sum'], 'num_985_sum': ['sum'],
        'num_100_sum': ['sum'], 'num_unq_sum': ['sum'], 'total_secs_sum': ['sum']}
processed_user_log = user_log.groupby(user_log.index).agg(func)
print(len(processed_user_log))
processed_user_log.columns = processed_user_log.columns.get_level_values(0)

processed_user_log.to_csv("../input/processed_user_log_v2.csv", index=False)
