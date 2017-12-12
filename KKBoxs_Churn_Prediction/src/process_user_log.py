import time
import pandas as pd


def process_user_log(chunk):
    grouped_object = chunk.groupby(chunk.index, sort=False)  # not sorting results in a minor speedup
    func = {'date': ['min', 'max', 'count'],
            'num_25': ['sum'], 'num_50': ['sum'],
            'num_75': ['sum'], 'num_985': ['sum'],
            'num_100': ['sum'], 'num_unq': ['sum'], 'total_secs': ['sum']}
    answer = grouped_object.agg(func)
    return answer


size = 4e7  # 1 million
reader = pd.read_csv('../input/user_logs.csv', chunksize=size, index_col=['msno'])
start_time = time.time()
for i in range(10):
    user_log_chunk = next(reader)
    if (i == 0):
        result = process_user_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del (user_log_chunk)

result.columns = ['_'.join(col).strip() for col in result.columns.values]

func = {'date_min': ['min'], 'date_max': ['max'], 'date_count': ['count'],
        'num_25_sum': ['sum'], 'num_50_sum': ['sum'],
        'num_75_sum': ['sum'], 'num_985_sum': ['sum'],
        'num_100_sum': ['sum'], 'num_unq_sum': ['sum'], 'total_secs_sum': ['sum']}
processed_user_log = result.groupby(result.index).agg(func)
print(len(processed_user_log))
processed_user_log.columns = processed_user_log.columns.get_level_values(0)

processed_user_log.to_csv("../input/processed_user_log_v1.csv")

size = 1e6
reader = pd.read_csv('../input/user_logs_v2.csv', chunksize=size, index_col=['msno'])
start_time = time.time()
for i in range(18):
    user_log_chunk = next(reader)
    if (i == 0):
        result = process_user_log(user_log_chunk)
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    else:
        result = result.append(process_user_log(user_log_chunk))
        print("Loop ", i, "took %s seconds" % (time.time() - start_time))
    del (user_log_chunk)

result.columns = ['_'.join(col).strip() for col in result.columns.values]

func = {'date_min': ['min'], 'date_max': ['max'], 'date_count': ['count'],
        'num_25_sum': ['sum'], 'num_50_sum': ['sum'],
        'num_75_sum': ['sum'], 'num_985_sum': ['sum'],
        'num_100_sum': ['sum'], 'num_unq_sum': ['sum'], 'total_secs_sum': ['sum']}
processed_user_log = result.groupby(result.index).agg(func)
print(len(processed_user_log))
processed_user_log.columns = processed_user_log.columns.get_level_values(0)

processed_user_log.to_csv("../input/processed_user_log_v2.csv")

user_log_v1 = pd.read_csv('../input/processed_user_log_v1.csv')
print(len(user_log_v1))
user_log_v2 = pd.read_csv('../input/processed_user_log_v2.csv')
print(len(user_log_v2))

user_log = user_log_v1.append(user_log_v2, ignore_index=True)

func = {'date_min': ['min'], 'date_max': ['max'], 'date_count': ['count'],
        'num_25_sum': ['sum'], 'num_50_sum': ['sum'],
        'num_75_sum': ['sum'], 'num_985_sum': ['sum'],
        'num_100_sum': ['sum'], 'num_unq_sum': ['sum'], 'total_secs_sum': ['sum']}
processed_user_log = user_log.groupby("msno").agg(func)
print(len(processed_user_log))
processed_user_log.columns = processed_user_log.columns.get_level_values(0)
print(processed_user_log.head())

processed_user_log.to_csv("../input/processed_user_log_all.csv")
